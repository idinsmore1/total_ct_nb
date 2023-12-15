import cupy as cp
import numpy as np
import kimimaro

from cupyx.scipy.ndimage import zoom
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from total_ct.vessels.vessel import Vessel

# Need to set np.bool = bool for kimimaro to work
np.bool = bool

class Centerline(Vessel):
    """Class used to create and represent a centerline within a vessel"""
    
    def create_centerline(self, kimimaro_const: int) -> cp.ndarray:
        """Primary method to create the centerline of the vessel
        :param kimimaro_const: the constant to use with kimimaro centerline generation, approximate average diameter of whatever vessel your creating a centerline for
        """
        if self.spacing[0] > 1 or self.spacing[0] < 0.75:
            self.vessel, self.vspacing = self._resample_vessel()
        else:
            self.vessel = self.seg
            self.vspacing = self.spacing
        # Generate the centerline
        raw_centerline = self._kimimaro_centerline(kimimaro_const)
        if raw_centerline is None:
            self.centerline = None
            return
        raw_centerline = self.evenly_sample_centerline(raw_centerline, int(len(raw_centerline) * 0.5))
        # smoothed_centerline = self.smooth_centerline(raw_centerline)
        # self.centerline = cp.asanyarray(raw_centerline)
        # Uniformly space the centerline points
        uniform_centerline = self.arclength_sample(raw_centerline)
        # Apply savitzky-golay filtering to smooth the centerline
        smoothed_centerline = self.savitzky_golay_filter(uniform_centerline)
        
        self.centerline = cp.asarray(smoothed_centerline)
        # Smooth the centerline with median filtering
        # window_size = raw_centerline.shape[0] // points
        # smoothed_centerline = self.smooth_centerline(raw_centerline)
        # self.centerline = cp.asarray(smoothed_centerline)
    
    def _kimimaro_centerline(self, kimimaro_const: int) -> cp.ndarray:
        """Internal Method to create a kimimaro centerline
        :param kimimaro_const: the constant to use with kimimaro centerline generation
        """
        skels = kimimaro.skeletonize(
            self.vessel.get(),
            teasar_params = {
                'const': kimimaro_const,
                'scale': 5
            },
            anisotropy=self.vspacing,
            progress=False
        )
        # If no centerline could be generated, return None
        if len(skels) == 0:
            return None
        # Get the vertices in array indicies
        vertices = skels[1].vertices / self.vspacing
        vertices = vertices.round().astype(np.uint16)
        # Order the vertices in connected-order 
        edges = skels[1].edges
        ordered = self.order_vertices(vertices, edges)
        return ordered
    
    
    @classmethod
    def order_vertices(cls, vertices, edges) -> np.ndarray:
        """Method to order the list of vertices from kimimaro
        :param vertices: the (N, 3) numpy array
        :param edges: the list of connective edges
        :return: The (N, 3) numpy array ordered by edges
        """
        # Step 1: Build the adjacency list
        adjacency_list = {}
        for edge in edges:
            if edge[0] not in adjacency_list:
                adjacency_list[edge[0]] = []
            if edge[1] not in adjacency_list:
                adjacency_list[edge[1]] = []

            adjacency_list[edge[0]].append(edge[1])
            adjacency_list[edge[1]].append(edge[0])

        # Step 2: Identify the start (and end) vertex
        start_vertex = None
        for vertex, connected in adjacency_list.items():
            if len(connected) == 1:
                start_vertex = vertex
                break

        # Sanity check
        if start_vertex is None:
            raise ValueError("A start vertex could not be found.")

        # Step 3: Traverse the graph from the start vertex
        ordered_vertices_indices = [start_vertex]
        current_vertex = start_vertex

        # Since we know the length of the path, we can loop N times
        for _ in range(len(vertices) - 1):
            # The next vertex will be the one that is not the previous
            for vertex in adjacency_list[current_vertex]:
                if vertex not in ordered_vertices_indices:
                    ordered_vertices_indices.append(vertex)
                    break
            current_vertex = ordered_vertices_indices[-1]

        # Step 4: Map the ordered indices to the original vertices
        ordered_vertices = [vertices[idx] for idx in ordered_vertices_indices]

        return np.asarray(ordered_vertices)
    
    @classmethod
    def smooth_centerline(cls, centerline, window_size=20):
        return median_filter(centerline, size=(window_size, 1))
    
    @classmethod
    def evenly_sample_centerline(cls, centerline, num_samples):
        # Calculate the cumulative distance along the centerline
        distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(centerline, axis=0), axis=1)])
        total_length = distances[-1]
        
        # Create an interpolation function for each dimension
        f_z = interp1d(distances, centerline[:, 0])
        f_y = interp1d(distances, centerline[:, 1])
        f_x = interp1d(distances, centerline[:, 2])
        
        # Evenly sample distances along the centerline
        sample_points = np.linspace(0, total_length, num_samples)
        # Sample the centerline
        sampled_centerline = np.column_stack((f_z(sample_points), f_y(sample_points), f_x(sample_points)))
        return sampled_centerline
    
    @classmethod
    def arclength_sample(cls, centerline: np.ndarray):
        """
        evenly space the centerline
        """
        differences = np.diff(centerline, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        arc_lengths = np.concatenate(([0], np.cumsum(distances)))
        total_length = arc_lengths[-1]
        normalized_arc_lengths = arc_lengths / total_length
        # Create interpolation functions for each coordinate
        interp_funcs = [interp1d(normalized_arc_lengths, centerline[:, i]) for i in range(3)]
        # Create a new uniform parameterization
        uniform_t = np.linspace(0, 1, len(centerline))
        # Interpolate the centerline points
        uniform_centerline = np.column_stack([f(uniform_t) for f in interp_funcs])
        return uniform_centerline

    @classmethod
    def savitzky_golay_filter(cls, uniform_centerline: np.ndarray, points: int=9, polyorder: int=2):
        """
        Apply Savitzky-Golay filtering to the uniform centerline for a smoother centerline
        Will help with perpendicular measurements later
        :param uniform_centerline: the output of arclength_sample
        :param points: The number of points to create the window, default=5
        :param polyorder: The order of the polynomial to be fit, default=2
        :return: the numpy array containing the smoothed centerline
        """
        window_length = uniform_centerline.shape[0] // points
        if window_length % 2 == 0:
            window_length += 1
        elif window_length == 1:
            window_length = 3
        smoothed_centerline = savgol_filter(uniform_centerline, window_length, polyorder, axis=0)
        return smoothed_centerline
    
    def _resample_vessel(self) -> list:
        """Internal method to resample segmentations with a slice thickness > 1mm
        to 1mm thickness, while maintaining x,y pixel spacing
        :return: a list containing [resampled_vessel, resampled_pixel_spacing]
        """
        # Set the new shape - number of z slices = z-slices * current z spacing 
        # Ex: 100 slices at 3mm thickness = 300 slices at 1mm thickness
        new_shape = (int(self.seg.shape[0] * self.spacing[0]), *self.seg.shape[1:])
        zoom_factors = [new / old for new, old in zip(new_shape, self.seg.shape)]
        new_vessel = zoom(self.seg, zoom_factors, order=1, mode='nearest')
        new_spacing = (1.0, *self.spacing[1:])
        return new_vessel, new_spacing

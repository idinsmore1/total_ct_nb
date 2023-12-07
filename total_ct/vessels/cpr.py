import cupy as cp
import numpy as np

from cucim.skimage.filters import gaussian
from cucim.skimage.measure import label, regionprops
from cucim.skimage.morphology import remove_small_holes
from cupyx.scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline

from total_ct.vessels.centerline import Centerline


class CurvedPlanarReformation(Centerline):
    """Class used to create and represent a straightened CPR of a vessel"""
    def __init__(self, orig_img: cp.ndarray, segmentation: cp.ndarray, spacing: list, task: str, vessel: str, kimimaro_const: int):
        # Initialize the centerline class
        super().__init__(orig_img, segmentation, spacing, task, vessel)
        self.kimimaro_const = kimimaro_const
        
        
    def create_straightened_cpr(self, grid_size: int=80) -> cp.ndarray:
        """Method to create the straightened CPR using the vessel and centerline
        :param grid_size: the length of the side of the sampling grid around the centerline in mm. Default=80
        :return: the straightened 3d volume
        """
        self.tangents = self._create_centerline_tangent_vectors()
        z, y, x = self.vspacing
        # Find the number of pixels to sample in each direction to get the desired size
        width_px = int(grid_size // x)
        height_px = int(grid_size // y)
        # Create an empty array to hold the cpr
        cpr = cp.zeros((int(len(self.centerline)), height_px, width_px))
        # Go through the centerline to create the cpr
        for i, (point, tangent) in enumerate(zip(self.centerline, self.tangents)):
            cpr[i, :, :] = self._sample_grid(self.vessel, point, tangent, width_px, height_px)
        self.cpr = cpr
    
    def measure_diameters(self) -> dict:
        """Method to measure the diameters for each of the slices of the straightened CPR
        :return: a dictionary containing the diameters, where each key is the CPR slice index measured
        """
        measurements = {}
        # Loop through the centerline
        for i in range(self.centerline.shape[0]):
            roi = self._smooth_cpr_slice(self.cpr[i])
            if roi is not None and roi.sum() > 0:
                # Measure the average, major, and minor axes
                diameters = measure_binary_diameter(roi, self.vspacing[1:])
                if diameters is not None:
                    avg_diam, major_diam, minor_diam = diameters
                    measurements[i] = {
                        'avg_diam': avg_diam,
                        'major_diam': major_diam,
                        'minor_diam': minor_diam,
                        'slice_idx': float(round(self.centerline[i][0], 1))
                    }
        self.diameters = measurements
        
    def find_max_diameter(self):
        """Method to find the max diameter of the vessel and it's neighboring values
        """
        max_diameter, max_key = self.get_max_diameter(self.diameters)
        comps = self.closest_measured_diameters(self.diameters, max_key)
        return max_diameter, comps
                
    # Internal methods
    def _create_centerline_tangent_vectors(self) -> cp.ndarray:
        """Method to create the tangent vectors for each point in the centerline"""
        centerline = self.centerline.get()
        s = np.zeros(len(centerline))
        # Find the spline points
        s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0)**2, axis=1)))
        # Generate the spline
        cs = CubicSpline(s, centerline, bc_type='natural')
        # Differentiate the spline to get the tangent vectors
        tangents = cs(s, 1)
        # Normalize the tangent vectors
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        return cp.asarray(tangents)
    
    @classmethod
    def _sample_grid(cls, image: cp.ndarray, center: cp.ndarray, tangent: cp.ndarray, width: int, height: int) -> cp.ndarray:
        """Internal method to sample points in a binary array around a centerline point and its tangent vector
        :param image: the segmentation the centerline was created from
        :param center: the point of the centerline to be sampled around
        :param tangent: the tangent vector for the centerline point
        :param width: the width of the grid in pixels to sample
        :param height: the height of the grid in pixels to sample
        :return: The 2d area sampled around the centerline
        """
        u, v = cls._calculate_perpendicular_vectors(tangent)
        x, y = cp.meshgrid(cp.linspace(-width/2, width/2, width), cp.linspace(-height/2, height/2, height))
        rectangle_points = center[:, None, None] +\
            x[None, :, :] * u[:, None, None] +\
            y[None, :, :] * v[:, None, None]
        rectangle_points = rectangle_points.reshape(3, -1)
        sampled_values = map_coordinates(image, rectangle_points, order=1, mode='nearest')
        return sampled_values.reshape((height, width))
        
    @classmethod
    def _calculate_perpendicular_vectors(cls, tangent: cp.ndarray) -> list:
        """Internal class method to create the normal and binormal vectors to a tangent vector
        :param tangent: the tangent vector to a centerline point
        :return: a list containing [normal_vector, binormal_vector]
        """
        # Calculate the normal vector
        u = cp.cross(tangent, cp.array([1, 0, 0]))
        # If the tangent is parallel to [1, 0, 0], adjust
        if cp.linalg.norm(u) < 1e-5:
            u = cp.cross(tangent, cp.array([0, 1, 0]))
        # Normalize the normal
        u = u / cp.linalg.norm(u)
        # Calculate the binormal
        v = cp.cross(tangent, u)
        v = v / cp.linalg.norm(v)
        return u, v
    
    @classmethod
    def _smooth_cpr_slice(cls, image: cp.ndarray):
        """Class method to fill any small gaps within the CPR slices and apply a gaussian filter to smooth the edges
        :param image: the slice of the curved planar reformation
        :return: smoothed image
        """
        roi = label(image)
        regions = regionprops(roi)
        if len(regions) > 1:
            roi = cls.closest_to_centroid(image, regions)
        elif len(regions) == 0:
            return None
        else:
            roi = image
        roi = remove_small_holes(roi.astype(cp.uint8))
        roi = gaussian(roi, sigma=2).round(0)
        return roi
    
    @classmethod
    def closest_to_centroid(cls, image, regions):
        """
        Function to find the region closest to the center centroid of an image, also filters out any non-continous segments
        """
        center_of_plane = cp.array(image.shape) / 2.0
        distances = [cp.linalg.norm(cp.array(region.centroid) - center_of_plane) for region in regions]
        min_distance_idx = int(cp.argmin(cp.asarray(distances)))
        center_region = regions[min_distance_idx]
        center_img = cp.zeros_like(image)
        center_img[center_region.coords[:, 0], center_region.coords[:, 1]] = 1
        return center_img
    
    @classmethod
    def get_max_diameter(cls, diameters):
        max_key = max(diameters, key=lambda x: diameters[x]['avg_diam'])
        return diameters[max_key], max_key

    @classmethod
    def closest_measured_diameters(cls, diameters, max_key, num_vals=4):
        key_list = list(diameters.keys())
        # Handle end of list exceptions
        if max_key == key_list[0]:
            comp_keys = key_list[:num_vals+1]
        elif max_key == key_list[-1]:
            comp_keys = key_list[-num_vals:]
        else:
            split = num_vals // 2
            for idx, val in enumerate(key_list):
                if val == max_key:
                    break
            # Handle split is too long for side exceptions
            if idx - split < 0:
                comp_keys = [key_list[i] for i in range(idx-split+1, idx+split+2)]
            elif idx + split > (len(key_list)-1):
                comp_keys = [key_list[i] for i in range(idx-split-1, idx+split)]
            else:
                comp_keys = [key_list[i] for i in range(idx-split, idx+split+1)]
        comp_dict = {}
        for key in comp_keys:
            diameter, slice_idx = diameters[key]['avg_diam'], round(diameters[key]['slice_idx'], 1)
            comp_dict[slice_idx] = diameter
        return comp_dict

    
###################### Generic functions that I don't want to assign to a class - could be used elsewhere ########
def measure_binary_diameter(array, pixel_spacing):
    """A function to measure the diameter of a binary object from a slice of a CT image
    :param array: the 2D slice containing the binary object
    :param pixel_spacing: the (y, x) spacing for the array in mm
    """
    labels = label(array)
    regions = regionprops(labels)
    if len(regions) == 0:
        return 0, 0, 0
    region = regions[0]
    major_endpoints = get_axis_endpoints(region.centroid, region.orientation, region.axis_major_length)
    minor_endpoints = get_axis_endpoints(region.centroid, region.orientation, region.axis_minor_length)
    major_mm = list(np.multiply(major_endpoints, pixel_spacing))
    minor_mm = list(np.multiply(minor_endpoints, pixel_spacing))
    major_dist = euclidean_distance(major_mm)
    minor_dist = euclidean_distance(minor_mm)
    return float(smooth_diameters(major_dist, minor_dist)), float(major_dist), float(minor_dist)
    

def get_axis_endpoints(centroid, orientation, axis_length):
    """generic function to get the endpoints of a regionprops axis
    :param centroid: the region.centroid
    :param orientation: the region.orientation
    :param axis_length: either region.axis_major_length or region.axis_minor_length
    """
    y0, x0 = centroid
    # calculate the endpoints of the major axis
    x1 = x0 - np.sin(orientation) * 0.5 * axis_length
    x2 = x0 + np.sin(orientation) * 0.5 * axis_length
    y1 = y0 - np.cos(orientation) * 0.5 * axis_length
    y2 = y0 + np.cos(orientation) * 0.5 * axis_length
    return (y1, x1), (y2, x2)
    

def euclidean_distance(points):
    """generic function to calculate the euclidean distance between a pair of points of arbitrary length
    :param points: a list of 2 points
    """
    p0, p1 = [*points]
    return round(np.sqrt(((p0 - p1) ** 2).sum()), 1)


def smooth_diameters(major_dist, minor_dist, threshold=10):
    """function to combine the major and minor axes distances, averaging if their difference is below a specific threshold.
    :param major_dist: the distance (typically in mm) of the major axis
    :param minor_dist: the distance (typically in mm) of the minor axis
    :param threshold: the cutoff threshold for averaging the axes' lengths. Default=15(mm)
    """
    if abs(major_dist - minor_dist) < threshold:
        return round((major_dist + minor_dist) / 2, 1)
    else:
        return min(major_dist, minor_dist)
    
    

# Functions to operate on arterial segmentations
import numpy as np
import cupy as cp
import kimimaro

from skimage.measure import label, regionprops
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter
from totalsegmentator.map_to_binary import class_map

# Set up global variables
inv_class_map = {v: k for k, v in class_map['total'].items()}
np.bool = bool

def measure_aorta(ct, device):
    aorta_stats = {}
    if ct.total_stats['aorta_volume_cm3'] == 0:
        print(f'Aorta not found in {ct.input}')
        ct.aorta_stats = aorta_stats
        return
    
    aorta = ct.total_seg == inv_class_map['aorta']
    spacing = ct.spacing
    vox_vol = ct.pixel_measurements['voxel_volume']
    midlines = ct.vertebrae_stats
    # Calculate the number of z-slices that would make up 5cm
    n_slices_5_cm = int(50 / spacing[2])
    # detect if the ct goes below the renal artery - were calling this the L1 midline
    abd_midlines = [midlines[f'vertebrae_L{i}']['midline'] for i in range(1, 6)]
    abd_midlines = [x for x in abd_midlines if x is not None]
    if abd_midlines:
        max_abd_midline = max(abd_midlines)
        # If the max_abd_midline is not at least 5 cm from the start of the CT, skip
        if max_abd_midline < n_slices_5_cm:
            max_abd_midline = None
    else:
        max_abd_midline = None
    # Get the T4 midline - if it exists, this is a good indicator that the entire thoracic aorta is in the image
    t4_midline = midlines['vertebrae_T4']['midline']
    # Get the regions of the thoracic aorta - if it doesn't exist at all we won't perform CPR
    thoracic_midlines = [midlines[f'vertebrae_T{i}']['midline'] for i in range(4, 11)]
    thoracic_midlines = [x for x in thoracic_midlines if x is not None]
    if thoracic_midlines:
        min_thor_midline = min(thoracic_midlines)
    else:
        min_thor_midline = None
    # Split the aorta
    # Abdominal aorta
    if max_abd_midline is not None:
        abdominal = aorta[:, :, :max_abd_midline]
    else:
        abdominal = None
    # Thoracic aorta - only perform if the descending and arch is in the image
    if t4_midline is not None and min_thor_midline is not None:
        thoracic = aorta[:, :, min_thor_midline:]
    else:
        thoracic = None
    
    if abdominal is not None:
        print('Measuring abdominal aorta diameters...')
    
        
        
def aorta_centerline_diameter(aorta, spacing, section):
    """function for measuring the centerline diameters of the aorta
    :param aorta: the numeric array containing the aorta
    :param spacing: the pixel spacing for the array
    :param section: the area of the aorta you are measuring ('abdominal', 'thoracic')
    """
    # change the axes of the aorta for easier centerline math
    aorta = aorta.transpose((2, 0, 1))
    aorta = aorta.astype(np.uint8)
    y, x, z = spacing
    spacing = [z, y, x]
    out_dict = {}
    # Make the centerline
    centerline = make_centerline(aorta, spacing, section)
    # If centerline fails, return missed
    if centerline is None:
        print(f'Centerline for {section} aorta could not be created.\n')
        out_dict['centerline_created'] = False
        return out_dict
    # Perform the curved planar reformation and straighten the segmentation
    reformat = reformat_to_centerline(aorta, centerline, spacing)
    return reformat
    

########################## Centerline Functions ###############################################  
def make_centerline(aorta, spacing, section) -> np.ndarray:
    """
    Make a centerline using kimimaro for the abdominal or thoracic aorta. Returns an (N, 3) array of ordered centerline points.
    :param aorta: the binary numpy array of shape (N, x, x) containing the aorta segmentation
    :param spacing: the pixel spacing for the segmentation
    :param section: the area of the aorta you are measuring ('abdominal', 'thoracic')
    """
    # Set the constant in mm for the aorta - average diameter approximation
    if section == 'thoracic':
        const = 35
    else:
        const = 25
    # perform the skeletonization
    skels = kimimaro.skeletonize(
        aorta,
        teasar_params={
            'const': const,
            'scale': 5,
        },
        anisotropy=spacing,
        progress=False
    )
    # if the skeleton is empty, return None
    if len(skels) == 0:
        return None
    # Get the vertices and edges
    vertices = skels[1].vertices / spacing
    vertices = vertices.round().astype(np.uint16)
    edges = skels[1].edges
    # Order the vertices in natural order
    ordered_vertices = order_vertices(vertices, edges)
    # Evenly space the vertices
    uniform_vertices = arclength_param(ordered_vertices)
    # Smooth the centerline
    window_length = uniform_vertices.shape[0] // 12
    smoothed_centerline = savitzky_golay_filtering(uniform_vertices, window_length, polyorder=2)
    return smoothed_centerline
    

def order_vertices(vertices, edges):
    """Function to order the list of vertices from kimimaro
    :param vertices: the (N, 3) numpy array
    :param edges: the list of connective edges
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


def arclength_param(centerline: np.ndarray):
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


def savitzky_golay_filtering(uniform_centerline, window_length, polyorder=2):
    """
    Apply Savitzky-Golay filtering to the uniform centerline for a smoother centerline
    Will help with perpendicular measurements later
    """
    window_length = min(window_length, len(uniform_centerline))
    if window_length % 2 == 0:
        window_length += 1
    smoothed_centerline = savgol_filter(uniform_centerline, window_length, polyorder, axis=0)
    return smoothed_centerline


########################## Reformation Functions ###############################################
def reformat_to_centerline(aorta, centerline, spacing, distance_threshold=2.0, grid_size=None):
    """
    Main function to take a centerline and a segmentation and return a straightened reformation.
    :param aorta: the binary numpy array holding the segmentation
    :param centerline: the output of make_centerline
    :param spacing: the pixel spacing of the segmentation
    """
    # Set the grid size to be 10cm based on the smallest x/y pixel spacing. cut in half for easier setup
    if grid_size is None:
        grid_size = int((80 // min(spacing[1:])) // 2)
    # Create the tangent vectors for each point on the centerline
    tangent_vectors = _create_tangent_vectors(centerline)
    # For each point in the centerline, using the tangent find seg-coords in the normal plane
    points_near_normal_planes = _find_nearby_points(aorta, centerline, tangent_vectors, distance_threshold)
    # reformat the image to straightened version
    reformatted = _map_to_new_coordinates(centerline, tangent_vectors, points_near_normal_planes, grid_size)
    return reformatted


def _create_tangent_vectors(centerline: np.ndarray) -> np.ndarray:
    """
    A function to fit a spline to the centerline and return an array of tangent vectors
    :param centerline: the output of make_centerline
    """
    points = centerline
    s = np.zeros(len(points))
    s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    # Fit a cubic spline to the data
    cs = CubicSpline(s, points, bc_type='natural')
    # Differentiate the spline to get the tangent vectors
    tangent_vectors = cs(s, 1)
    # Normalize the tangent vectors
    tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, None]
    return tangent_vectors


def _find_nearby_points(aorta: np.ndarray, centerline: np.ndarray, tangent_vectors: np.ndarray, distance_threshold=3.0):
    """
    a function to find all points of the segmentation within the normal plane of the tangent and centerline
    :param aorta: the binary array that holds the segmentation
    :param centerline: the output of make_centerline
    :param tangent_vectors: the result of _create_tangent_vectors
    :param distance_threshold: the acceptable distance between the plane and point to be included. Default=2.0
    """
    seg_points = np.argwhere(aorta == 1)
    points_near_normal_planes = []
    for i in range(len(centerline)):
        # Calculate distances from segmentation points to the current normal plane
        distances = np.abs(np.dot(seg_points - centerline[i], tangent_vectors[i])) / np.linalg.norm(tangent_vectors[i])
        # Find points that are close to the normal plane
        close_points = seg_points[distances < distance_threshold]
        # Add them to the list
        points_near_normal_planes.append(close_points)
    return points_near_normal_planes


def _map_to_new_coordinates(centerline, tangent_vectors, points_near_normal_planes, grid_size):
    """
    a function to reformat the segmentation to a straightened view, so that only the diameters vary
    :param centerline: the output of make_centerline
    :param tangent_vectors: the output of _create_tangent_vectors
    :param points_near_normal_planes: the output of _find_nearby_points
    :param grid_size: the desired output grid size in mm
    """
    # Set up a local coordinate system
    local_coordinate_systems = []
    for tangent in tangent_vectors:
        z_axis = tangent
        # Set a non-parallel axis to start
        x_axis = np.cross(z_axis, [1, 0, 0])  
        if np.allclose(x_axis, 0):
            x_axis = np.cross(z_axis, [0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)

        # Normalize the axes
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)

        local_coordinate_systems.append((x_axis, y_axis, z_axis))
    # Map points near planes
    mapped_points = []
    for i, (x_axis, y_axis, z_axis) in enumerate(local_coordinate_systems):
        transformation_matrix = np.array([x_axis, y_axis, z_axis]).T
        local_coords = np.dot(points_near_normal_planes[i] - centerline[i], transformation_matrix)
        mapped_points.append(local_coords)
    # Reformat the image
    reformatted_images = np.zeros((len(centerline), grid_size * 2, grid_size * 2))
    for i, local_coords in enumerate(mapped_points):
        # Convert local coordinates to pixel indices
        pixel_indices_x = np.round((local_coords[:, 0] + grid_size) / (2 * grid_size) * grid_size * 2).astype(int)
        pixel_indices_y = np.round((local_coords[:, 1] + grid_size) / (2 * grid_size) * grid_size * 2).astype(int)

        # Filter out-of-bounds indices
        valid_indices = (pixel_indices_x >= 0) & (pixel_indices_x < grid_size * 2) & (pixel_indices_y >= 0) & (pixel_indices_y < grid_size * 2)
        pixel_indices_x = pixel_indices_x[valid_indices]
        pixel_indices_y = pixel_indices_y[valid_indices]
        # Set the pixel values at the indices to 1 (or any other value you'd like)
        reformatted_images[i, pixel_indices_y, pixel_indices_x] = 1
    return reformatted_images
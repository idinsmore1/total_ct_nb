# Functions to operate on arterial segmentations
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import kimimaro

from cupyx.scipy.ndimage import zoom, map_coordinates
from cucim.skimage.filters import gaussian
from cucim.skimage.measure import label, regionprops
from cucim.skimage.morphology import remove_small_holes
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
# from scipy.ndimage import map_coordinates
from totalsegmentator.map_to_binary import class_map

from total_ct.utils import block_print

# Set up global variables
inv_class_map = {v: k for k, v in class_map['total'].items()}
np.bool = bool

def measure_aorta(ct):
    aorta_stats = {}
    if ct.total_stats['aorta_volume_cm3'] == 0:
        print(f'Aorta not found in {ct.input}')
        ct.aorta_stats = aorta_stats
        return
    # Set up necessary variables
    aorta = ct.total_seg == inv_class_map['aorta']
    spacing = ct.spacing
    vox_vol = ct.pixel_measurements['voxel_volume']
    midlines = ct.vertebrae_stats
    out_dict = {}
    abdominal_dict = {}
    thoracic_dict = {}
    # Calculate the number of z-slices that would make up 5cm
    n_slices_5_cm = int(50 / spacing[2])
    # detect if the ct goes below the renal artery - were calling this the L1 midline
    abd_midlines = [midlines[f'vertebrae_L{i}']['midline'] for i in range(1, 6)]
    abd_midlines = [x for x in abd_midlines if x is not None]
    if abd_midlines:
        max_abd_midline = max(abd_midlines)
        # If the max_abd_midline is not at least 5 cm from the start of the CT, skip
        if max_abd_midline < n_slices_5_cm:
            print(f'Abdominal aorta segmentation is less than 50mm long: {int(max_abd_midline * spacing[2])}')
            max_abd_midline = None
    else:
        max_abd_midline = None
    # Get the T4 midline - if it exists, this is a good indicator that the entire thoracic aorta is in the image
    t4_midline = midlines['vertebrae_T4']['midline']
    # Get the regions of the thoracic aorta - if it doesn't exist at all we won't perform CPR
    thoracic_midlines = [*[midlines[f'vertebrae_T{i}']['midline'] for i in range(1, 13)], midlines[f'vertebrae_L1']['midline']]
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
        # thoracic = aorta[:, :, min_thor_midline:]
        thoracic = aorta
    else:
        thoracic = None
    
    if abdominal is not None:
        print('Measuring abdominal aorta diameters...')
        abdominal_dict = {}
        abdominal_max, abdominal_comps = aorta_centerline_diameter(abdominal, spacing, 'abdominal')
        for key in ['avg_diam', 'major_diam', 'minor_diam', 'slice_idx', 'comp']:
            if key != 'comp':
                abdominal_dict[f'abd_aorta_{key}'] = abdominal_max[key]
            else:
                abdominal_dict[f'abd_aorta_{key}'] = abdominal_comps
    if thoracic is not None:
        print('Measuring thoracic aorta diameters...')
        thoracic_dict = {}
        asc_max, asc_comps, desc_max, desc_comps = aorta_centerline_diameter(thoracic, spacing, 'thoracic')
        for name, pair in zip(['asc', 'desc'], [(asc_max, asc_comps), (desc_max, desc_comps)]):
            max_data, comp = pair
            for key in ['avg_diam', 'major_diam', 'minor_diam', 'slice_idx', 'comp']:
                if key != 'comp':
                    thoracic_dict[f'{name}_aorta_{key}'] = max_data[key]
                else:
                    thoracic_dict[f'{name}_aorta_{key}'] = comp

    out_dict.update(abdominal_dict)
    out_dict.update(thoracic_dict)
    return out_dict
                    
def aorta_centerline_diameter(aorta, spacing, section):
    """function for measuring the centerline diameters of the aorta
    :param aorta: the numeric array containing the aorta
    :param spacing: the pixel spacing for the array
    :param section: the area of the aorta you are measuring ('abdominal', 'thoracic')
    """
    # change the axes of the aorta for easier centerline math 
    aorta = aorta.transpose((2, 0, 1))
    aorta = cp.flip(aorta, axis=(0, 2))
    aorta = aorta.astype(np.uint8)
    plt.imshow(aorta.get().max(1))
    plt.show()
    y, x, z = spacing
    spacing = [z, y, x]
    out_dict = {}
    if z > 1:
        print(f'Reformatting {section} to 1mm slice thickness')
        aorta, spacing = resample_aorta(aorta, spacing)
    print(f'{spacing=}')
    # Make the centerline
    centerline = make_centerline(aorta, spacing, section)
    # If centerline fails, return missed
    if centerline is None:
        print(f'Centerline for {section} aorta could not be created.\n')
        out_dict['centerline_created'] = False
        return out_dict
    # Perform the curved planar reformation and straighten the segmentation
    print('Making Curved Planar Reformation...')
    cpr = curved_planar_reformation(aorta, centerline, spacing)
    plt.imshow(cpr.get().max(1))
    plt.show()
    # Measure diameters
    # print(f'Measuring {section} diameters...')
    measurements = cpr_diameters(aorta, cpr, spacing, centerline, section)
    return measurements
    

########################## Centerline Functions ###############################################
def resample_aorta(aorta, spacing):
    """Function to resample the aorta segmentation to 1x1x1mm 
    :param aorta: the binary array containing the aorta
    :param spacing: the original pixel spacing for the array
    """
    new_shape = (aorta.shape[0] * spacing[0], *aorta.shape[1:])
    print(f'{new_shape=}')
    zoom_factors = [new / old for new, old in zip(new_shape, aorta.shape)]
    new_aorta = zoom(aorta, zoom_factors, order=1, mode='nearest')
    new_spacing = [1.0, *spacing[1:]]
    return new_aorta, new_spacing


def make_centerline(aorta, spacing, section) -> np.ndarray:
    """
    Make a centerline using kimimaro for the abdominal or thoracic aorta. Returns an (N, 3) array of ordered centerline points.
    :param aorta: the binary numpy array of shape (N, x, x) containing the aorta segmentation
    :param spacing: the pixel spacing for the segmentation
    :param section: the area of the aorta you are measuring ('abdominal', 'thoracic')
    """
    # Set the constant in mm for the aorta - average diameter approximation
    if section == 'thoracic':
        const = 30
    else:
        const = 25
    # perform the skeletonization
    skels = kimimaro.skeletonize(
        aorta.get(),
        teasar_params={
            'const': const,
            'scale': 5,
        },
        anisotropy=spacing,
        progress=True
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
    window_length = uniform_vertices.shape[0] // 5
    # We want the window length to be at least 3
    if window_length == 1:
        window_length = 3
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
def curved_planar_reformation(aorta, centerline, spacing, grid_size=80):
    """
    Main function to take a centerline and a segmentation and return a straightened reformation.
    :param aorta: the binary numpy array holding the segmentation
    :param centerline: the output of make_centerline
    :param spacing: the pixel spacing of the segmentation
    :param grid_size: the size of the grid(mm) to sample around the centerline. Default = 80
    """
    # Create the tangent vectors for the centerline
    tangents = create_centerline_tangent_vectors(centerline)
    z, y, x = spacing
    # Set the number of pixels to sample from the aorta segmentation
    width = int(grid_size // x)
    height = int(grid_size // y)
    # Create an empty array to hold the CPR
    cpr = cp.zeros((len(centerline), height, width))
    for i, (point, tangent) in enumerate(zip(centerline, tangents)):
        cpr[i, :, :] = sample_rectangle(aorta, point, tangent, width, height)
    return cpr


def create_centerline_tangent_vectors(centerline: np.ndarray) -> np.ndarray:
    """Function to create the centerline tangents
    :param centerline: The centerline of the vessel
    """
    s = np.zeros(len(centerline))
    # Find the spline points
    s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0)**2, axis=1)))
    # Generate the spline
    cs = CubicSpline(s, centerline, bc_type='natural')
    # Differentiate the spline to get the tangent vectors
    tangents = cs(s, 1)
    # Normalize the tangent vectors
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return tangents


# Find the perpendicular vectors
def calculate_pependicular_vectors(tangent):
    u = np.cross(tangent, np.array([1, 0, 0]))
    # If the tangent is parallel, adjust
    if np.linalg.norm(u) < 1e-5:
        u = np.cross(tangent, np.array([0, 1, 0]))
    # Normalize vector
    u = u / np.linalg.norm(u)
    # Calculate other perpendicular vector
    v = np.cross(tangent, u)
    v = v / np.linalg.norm(v)
    return u, v

# rectangle sampling - i think this works better
def sample_rectangle(image, center, tangent, width, height):
    u, v = calculate_pependicular_vectors(tangent)
    # num_pixels_x = int(width / pixel_spacing_x)
    # num_pixels_y = int(height / pixel_spacing_y)
    x, y = np.meshgrid(np.linspace(-width/2, width/2, width), np.linspace(-height/2, height/2, height))
    rectangle_points = center[:, None, None] + x[None, :, :] * u[:, None, None] + y[None, :, :] * v[:, None, None]
    rectangle_points = cp.asarray(rectangle_points.reshape(3, -1))
    sampled_values = map_coordinates(image, rectangle_points, order=1, mode='nearest')
    return sampled_values.reshape((height, width))


########################## Measurement Functions ###############################################
def cpr_diameters(aorta, cpr, spacing, centerline, section) -> dict:
    """Function to measure the diameters of the straightened aorta. If measuring the thoracic aorta, will return the ascending and descending measurements
    :param aorta: the array containing the aorta segmentation - used if measuring the thoracic aorta to detect the aortic arch
    :param cpr: the curved planar reformation image
    :param spacing: the pixel spacing of the cpr
    :param centerline: the centerline used to generate the cpr
    :param section: either "abdominal" or "thoracic"
    """
    if section == 'abdominal':
        split = 0
        abd_diams =  measure_diameters(cpr, centerline, spacing, split)
        max_diam, max_key = get_max_diameter(abd_diams)
        abd_comps = closest_measured_diameters(abd_diams, max_key)
        return max_diam, abd_comps
    else:
        asc_cpr, asc_centerline, desc_cpr, desc_centerline, split = split_thoracic_aorta(aorta, cpr, centerline)
        # Ascending measurements
        asc_diams = measure_diameters(asc_cpr, asc_centerline, spacing, split)
        max_asc_diam, max_asc_key = get_max_diameter(asc_diams)
        asc_comps = closest_measured_diameters(asc_diams, max_asc_key)
        # Descending measurments
        desc_diams = measure_diameters(desc_cpr, desc_centerline, spacing, split) 
        max_desc_diam, max_desc_key = get_max_diameter(desc_diams)
        desc_comps = closest_measured_diameters(desc_diams, max_desc_key)
        return max_asc_diam, asc_comps, max_desc_diam, desc_comps
    
                                               
def split_thoracic_aorta(aorta, cpr, centerline):
    arch_peak = round(np.argmin(centerline, axis=0)[0])
    asc_cpr = cpr[:arch_peak, :, :]
    asc_centerline = centerline[:arch_peak]
    desc_cpr = cpr[arch_peak:, :, :]
    desc_centerline = centerline[arch_peak:]
    split = arch_endpoint_idx(aorta)
    return asc_cpr, asc_centerline, desc_cpr, desc_centerline, split


def arch_endpoint_idx(aorta):
    split = 0
    for i in range(aorta.shape[0]):
        labels = label(aorta[i])
        regions = regionprops(labels)
        if len(regions) == 2:
            split = i
            break
    return split


def measure_diameters(cpr, centerline, spacing, split):
    """Function to take a curved planar reformation, centerline, and spacing and measure relevant diameters
    :param cpr: the curved planar reformation array
    :param centerline: the centerline from the original image
    :param spacing: the pixel spacing of the cpr
    :param split: If measuring the thoracic aorta, this is the break point for the aortic arch. Should be 0 for abdominal aorta
    """
    measurements = {}
    for i in range(centerline.shape[0]):
        if centerline[i][0] >= split:
            roi = smooth_cpr_slice(cpr[i])
            if roi is not None and roi.sum() > 0:
                diameters = measure_binary_diameter(roi, spacing[1:])
                if diameters is not None:
                    avg_diam, major_diam, minor_diam = diameters
                    measurements[i] = {
                        'avg_diam': float(avg_diam),
                        'major_diam': float(major_diam),
                        'minor_diam': float(minor_diam),
                        'slice_idx': float(round(centerline[i][0], 1))
                    }
    return measurements
            
            
def smooth_cpr_slice(image):
    """function to fill any small gaps within the CPR slices and apply a gaussian filter to smooth the edges
    :param image: an axial slice of the curved planar reformation
    :return smoothed image
    """
    roi = label(image)
    regions = regionprops(roi)
    if len(regions) > 1:
        roi = closest_to_centroid(image, regions)
    elif len(regions) == 0:
        return None
    else:
        roi = image
    roi = remove_small_holes(roi.astype(np.uint8))
    roi = gaussian(roi, sigma=2).round(0)
    return roi


def closest_to_centroid(image, regions):
    """
    Function to find the region closest to the center centroid of an image, also filters out any non-continous segments
    """
    center_of_plane = np.array(image.shape) / 2.0
    distances = [np.linalg.norm(np.array(region.centroid) - center_of_plane) for region in regions]
    min_distance_idx = np.argmin(distances)
    center_region = regions[min_distance_idx]
    center_img = np.zeros_like(image)
    center_img[center_region.coords[:, 0], center_region.coords[:, 1]] = 1
    return center_img


def get_max_diameter(diameters):
    max_key = max(diameters, key=lambda x: diameters[x]['avg_diam'])
    return diameters[max_key], max_key


def closest_measured_diameters(diameters, max_key, num_vals=4):
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
    return smooth_diameters(major_dist, minor_dist), major_dist, minor_dist 
    

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
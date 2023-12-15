import nibabel as nib
import cupy as cp
import numpy as np
from cupyx.scipy import ndimage
from cucim.skimage.transform import resize
import importlib

from totalsegmentator.resampling import resample_img, resample_img_cucim
# Utility functions - can be for random one offs that may need to be repeated
cupy_available = importlib.util.find_spec("cupy") is not None
cucim_available = importlib.util.find_spec("cucim") is not None

def block_print(*args):
    """function that pretty prints a list of statements in a block. Useful if you want multiple print statements without writing print a bunch of times
    :param statements: all the text strings you want to print
    """
    for statement in args:
        if statement == args[-1]:
            statement = f'{statement}\n'
        print(statement)
        

def combine_dictionaries(list_of_dicts):
    holder_dict = {}
    for dict_ in list_of_dicts:
        holder_dict.update(dict_)
    return holder_dict

def prefix_dict_keys(dictionary, prefix):
    """Function to prefix the keys of a dictionary. Useful for final outputs
    :param dictionary: the dictionary to replace key names
    :param prefix: the prefix desired to be added to the keys
    """
    return {f'{prefix}_{k}': v for k, v in dictionary.items()}


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1, dtype=None, remove_negative=False, force_affine=None):
    """
    Resample nifti image to the new spacing (uses resample_img() internally).
    
    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    dtype: output datatype
    remove_negative: set all negative values to 0. Useful if resampling introduced negative values.
    force_affine: if you pass an affine then this will be used for the output image (useful if you have to make sure
                  that the resampled has identical affine to some other image. In this case also set target_shape.)

    Works for 2D and 3D and 4D images.

    If downsampling an image and then upsampling again to original resolution the resulting image can have
    a shape which is +-1 compared to original shape, because of rounding of the shape to int.
    To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
    spacing will be calculated which is needed to get to target_shape.
    In this case however the calculated spacing can be slighlty different from the desired new_spacing. This will
    result in a slightly different affine. To avoid this the desired affine can be writen by force with "force_affine".

    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    data =  img_in.dataobj[..., :].astype(np.float32) # quite slow
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if len(img_spacing) == 4:
        img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

    if type(new_spacing) is float:
        new_spacing = [new_spacing,] * 3   # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(list(img_spacing) + [new_spacing[2],])

    if target_shape is not None:
        # Find the right zoom to exactly reach the target_shape.
        # We also have to adapt the spacing to this new zoom.
        zooms = np.array(target_shape) / old_shape  
        new_spacing = img_spacing / zooms  
    else:
        zooms = img_spacing / new_spacing
    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)
    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zooms[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zooms[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zooms[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))


    if cupy_available and cucim_available:
        new_data = resample_img_cucim(data, zoom=zooms, order=order, nr_cpus=nr_cpus)  # gpu resampling is being odd - isn't loading values properly
    else:
        new_data = resample_img(data, zoom=zooms, order=order, nr_cpus=nr_cpus)  # cpu resampling
    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)
    if force_affine is not None:
        new_affine = force_affine
        
    new_nifti = nib.Nifti1Image(new_data, affine=new_affine)
    return new_nifti

final_df_columns = [
    'mrn',
    'accession',
    'series_name',
    'ct_direction',
    'image_type',
    'sex',
    'birthday',
    'scan_date',
    'age_at_scan',
    'length_mm',
    'width_mm',
    'slice_thickness_mm',
    'manufacturer',
    'manufacturer_model',
    'kvp',
    'sequence_name',
    'protocol_name',
    'contrast_bolus_agent',
    'contrast_bolus_route',
    'multienergy_ct',
    'spacing',
    'total_body_volume',
    'total_abdominal_body_volume',
    'total_subcutaneous_fat_cm3',
    'total_abdominal_subcutaneous_fat_cm3',
    'total_torso_fat_cm3',
    'total_abdominal_torso_fat_cm3',
    'total_skeletal_muscle_cm3',
    'total_abdominal_skeletal_muscle_cm3',
    'L1_body_area_cm2',
    'L1_body_circ_cm',
    'L1_body_circ_in',
    'L1_subcutaneous_fat_cm2',
    'L1_torso_fat_cm2',
    'L1_skeletal_muscle_cm2',
    'L3_body_area_cm2',
    'L3_body_circ_cm',
    'L3_body_circ_in',
    'L3_subcutaneous_fat_cm2',
    'L3_torso_fat_cm2',
    'L3_skeletal_muscle_cm2',
    'L5_body_area_cm2',
    'L5_body_circ_cm',
    'L5_body_circ_in',
    'L5_subcutaneous_fat_cm2',
    'L5_torso_fat_cm2',
    'L5_skeletal_muscle_cm2',
    'vertebrae_S1_midline',
    'vertebrae_S1_average_axis_mm',
    'vertebrae_S1_major_axis_diameter_mm',
    'vertebrae_S1_minor_axis_diameter_mm',
    'vertebrae_L5_midline',
    'vertebrae_L5_average_axis_mm',
    'vertebrae_L5_major_axis_diameter_mm',
    'vertebrae_L5_minor_axis_diameter_mm',
    'vertebrae_L4_midline',
    'vertebrae_L4_average_axis_mm',
    'vertebrae_L4_major_axis_diameter_mm',
    'vertebrae_L4_minor_axis_diameter_mm',
    'vertebrae_L3_midline',
    'vertebrae_L3_average_axis_mm',
    'vertebrae_L3_major_axis_diameter_mm',
    'vertebrae_L3_minor_axis_diameter_mm',
    'vertebrae_L2_midline',
    'vertebrae_L2_average_axis_mm',
    'vertebrae_L2_major_axis_diameter_mm',
    'vertebrae_L2_minor_axis_diameter_mm',
    'vertebrae_L1_midline',
    'vertebrae_L1_average_axis_mm',
    'vertebrae_L1_major_axis_diameter_mm',
    'vertebrae_L1_minor_axis_diameter_mm',
    'vertebrae_T12_midline',
    'vertebrae_T12_average_axis_mm',
    'vertebrae_T12_major_axis_diameter_mm',
    'vertebrae_T12_minor_axis_diameter_mm',
    'vertebrae_T11_midline',
    'vertebrae_T11_average_axis_mm',
    'vertebrae_T11_major_axis_diameter_mm',
    'vertebrae_T11_minor_axis_diameter_mm',
    'vertebrae_T10_midline',
    'vertebrae_T10_average_axis_mm',
    'vertebrae_T10_major_axis_diameter_mm',
    'vertebrae_T10_minor_axis_diameter_mm',
    'vertebrae_T9_midline',
    'vertebrae_T9_average_axis_mm',
    'vertebrae_T9_major_axis_diameter_mm',
    'vertebrae_T9_minor_axis_diameter_mm',
    'vertebrae_T8_midline',
    'vertebrae_T8_average_axis_mm',
    'vertebrae_T8_major_axis_diameter_mm',
    'vertebrae_T8_minor_axis_diameter_mm',
    'vertebrae_T7_midline',
    'vertebrae_T7_average_axis_mm',
    'vertebrae_T7_major_axis_diameter_mm',
    'vertebrae_T7_minor_axis_diameter_mm',
    'vertebrae_T6_midline',
    'vertebrae_T6_average_axis_mm',
    'vertebrae_T6_major_axis_diameter_mm',
    'vertebrae_T6_minor_axis_diameter_mm',
    'vertebrae_T5_midline',
    'vertebrae_T5_average_axis_mm',
    'vertebrae_T5_major_axis_diameter_mm',
    'vertebrae_T5_minor_axis_diameter_mm',
    'vertebrae_T4_midline',
    'vertebrae_T4_average_axis_mm',
    'vertebrae_T4_major_axis_diameter_mm',
    'vertebrae_T4_minor_axis_diameter_mm',
    'vertebrae_T3_midline',
    'vertebrae_T3_average_axis_mm',
    'vertebrae_T3_major_axis_diameter_mm',
    'vertebrae_T3_minor_axis_diameter_mm',
    'vertebrae_T2_midline',
    'vertebrae_T2_average_axis_mm',
    'vertebrae_T2_major_axis_diameter_mm',
    'vertebrae_T2_minor_axis_diameter_mm',
    'vertebrae_T1_midline',
    'vertebrae_T1_average_axis_mm',
    'vertebrae_T1_major_axis_diameter_mm',
    'vertebrae_T1_minor_axis_diameter_mm',
    'vertebrae_C7_midline',
    'vertebrae_C7_average_axis_mm',
    'vertebrae_C7_major_axis_diameter_mm',
    'vertebrae_C7_minor_axis_diameter_mm',
    'vertebrae_C6_midline',
    'vertebrae_C6_average_axis_mm',
    'vertebrae_C6_major_axis_diameter_mm',
    'vertebrae_C6_minor_axis_diameter_mm',
    'vertebrae_C5_midline',
    'vertebrae_C5_average_axis_mm',
    'vertebrae_C5_major_axis_diameter_mm',
    'vertebrae_C5_minor_axis_diameter_mm',
    'vertebrae_C4_midline',
    'vertebrae_C4_average_axis_mm',
    'vertebrae_C4_major_axis_diameter_mm',
    'vertebrae_C4_minor_axis_diameter_mm',
    'vertebrae_C3_midline',
    'vertebrae_C3_average_axis_mm',
    'vertebrae_C3_major_axis_diameter_mm',
    'vertebrae_C3_minor_axis_diameter_mm',
    'vertebrae_C2_midline',
    'vertebrae_C2_average_axis_mm',
    'vertebrae_C2_major_axis_diameter_mm',
    'vertebrae_C2_minor_axis_diameter_mm',
    'vertebrae_C1_midline',
    'vertebrae_C1_average_axis_mm',
    'vertebrae_C1_major_axis_diameter_mm',
    'vertebrae_C1_minor_axis_diameter_mm',
    'abd_aorta_calc_mm3',
    'abd_aorta_volume_mm3',
    'abd_aorta_avg_diam',
    'abd_aorta_major_diam',
    'abd_aorta_minor_diam',
    'abd_aorta_slice_idx',
    'abd_aorta_comp_diams',
    'thor_aorta_calc_mm3',
    'thor_aorta_volume_mm3',
    'asc_aorta_avg_diam',
    'asc_aorta_major_diam',
    'asc_aorta_minor_diam',
    'asc_aorta_slice_idx',
    'asc_aorta_comp_diams',
    'desc_aorta_avg_diam',
    'desc_aorta_major_diam',
    'desc_aorta_minor_diam',
    'desc_aorta_slice_idx',
    'desc_aorta_comp_diams',
    'spleen_volume_cm3',
    'spleen_average_HU',
    'kidney_right_volume_cm3',
    'kidney_right_average_HU',
    'kidney_left_volume_cm3',
    'kidney_left_average_HU',
    'gallbladder_volume_cm3',
    'gallbladder_average_HU',
    'liver_volume_cm3',
    'liver_average_HU',
    'stomach_volume_cm3',
    'stomach_average_HU',
    'pancreas_volume_cm3',
    'pancreas_average_HU',
    'adrenal_gland_right_volume_cm3',
    'adrenal_gland_right_average_HU',
    'adrenal_gland_left_volume_cm3',
    'adrenal_gland_left_average_HU',
    'lung_upper_lobe_left_volume_cm3',
    'lung_upper_lobe_left_average_HU',
    'lung_lower_lobe_left_volume_cm3',
    'lung_lower_lobe_left_average_HU',
    'lung_upper_lobe_right_volume_cm3',
    'lung_upper_lobe_right_average_HU',
    'lung_middle_lobe_right_volume_cm3',
    'lung_middle_lobe_right_average_HU',
    'lung_lower_lobe_right_volume_cm3',
    'lung_lower_lobe_right_average_HU',
    'esophagus_volume_cm3',
    'esophagus_average_HU',
    'trachea_volume_cm3',
    'trachea_average_HU',
    'thyroid_gland_volume_cm3',
    'thyroid_gland_average_HU',
    'small_bowel_volume_cm3',
    'small_bowel_average_HU',
    'duodenum_volume_cm3',
    'duodenum_average_HU',
    'colon_volume_cm3',
    'colon_average_HU',
    'urinary_bladder_volume_cm3',
    'urinary_bladder_average_HU',
    'prostate_volume_cm3',
    'prostate_average_HU',
    'kidney_cyst_left_volume_cm3',
    'kidney_cyst_left_average_HU',
    'kidney_cyst_right_volume_cm3',
    'kidney_cyst_right_average_HU',
    # Removing vertebrae from output - not really interesting
    # 'sacrum_volume_cm3',
    # 'sacrum_average_HU',
    # 'vertebrae_S1_volume_cm3',
    # 'vertebrae_S1_average_HU',
    # 'vertebrae_L5_volume_cm3',
    # 'vertebrae_L5_average_HU',
    # 'vertebrae_L4_volume_cm3',
    # 'vertebrae_L4_average_HU',
    # 'vertebrae_L3_volume_cm3',
    # 'vertebrae_L3_average_HU',
    # 'vertebrae_L2_volume_cm3',
    # 'vertebrae_L2_average_HU',
    # 'vertebrae_L1_volume_cm3',
    # 'vertebrae_L1_average_HU',
    # 'vertebrae_T12_volume_cm3',
    # 'vertebrae_T12_average_HU',
    # 'vertebrae_T11_volume_cm3',
    # 'vertebrae_T11_average_HU',
    # 'vertebrae_T10_volume_cm3',
    # 'vertebrae_T10_average_HU',
    # 'vertebrae_T9_volume_cm3',
    # 'vertebrae_T9_average_HU',
    # 'vertebrae_T8_volume_cm3',
    # 'vertebrae_T8_average_HU',
    # 'vertebrae_T7_volume_cm3',
    # 'vertebrae_T7_average_HU',
    # 'vertebrae_T6_volume_cm3',
    # 'vertebrae_T6_average_HU',
    # 'vertebrae_T5_volume_cm3',
    # 'vertebrae_T5_average_HU',
    # 'vertebrae_T4_volume_cm3',
    # 'vertebrae_T4_average_HU',
    # 'vertebrae_T3_volume_cm3',
    # 'vertebrae_T3_average_HU',
    # 'vertebrae_T2_volume_cm3',
    # 'vertebrae_T2_average_HU',
    # 'vertebrae_T1_volume_cm3',
    # 'vertebrae_T1_average_HU',
    # 'vertebrae_C7_volume_cm3',
    # 'vertebrae_C7_average_HU',
    # 'vertebrae_C6_volume_cm3',
    # 'vertebrae_C6_average_HU',
    # 'vertebrae_C5_volume_cm3',
    # 'vertebrae_C5_average_HU',
    # 'vertebrae_C4_volume_cm3',
    # 'vertebrae_C4_average_HU',
    # 'vertebrae_C3_volume_cm3',
    # 'vertebrae_C3_average_HU',
    # 'vertebrae_C2_volume_cm3',
    # 'vertebrae_C2_average_HU',
    # 'vertebrae_C1_volume_cm3',
    # 'vertebrae_C1_average_HU',
    'heart_volume_cm3',
    'heart_average_HU',
    'aorta_volume_cm3',
    'aorta_average_HU',
    'pulmonary_vein_volume_cm3',
    'pulmonary_vein_average_HU',
    'brachiocephalic_trunk_volume_cm3',
    'brachiocephalic_trunk_average_HU',
    'subclavian_artery_right_volume_cm3',
    'subclavian_artery_right_average_HU',
    'subclavian_artery_left_volume_cm3',
    'subclavian_artery_left_average_HU',
    'common_carotid_artery_right_volume_cm3',
    'common_carotid_artery_right_average_HU',
    'common_carotid_artery_left_volume_cm3',
    'common_carotid_artery_left_average_HU',
    'brachiocephalic_vein_left_volume_cm3',
    'brachiocephalic_vein_left_average_HU',
    'brachiocephalic_vein_right_volume_cm3',
    'brachiocephalic_vein_right_average_HU',
    'atrial_appendage_left_volume_cm3',
    'atrial_appendage_left_average_HU',
    'superior_vena_cava_volume_cm3',
    'superior_vena_cava_average_HU',
    'inferior_vena_cava_volume_cm3',
    'inferior_vena_cava_average_HU',
    'portal_vein_and_splenic_vein_volume_cm3',
    'portal_vein_and_splenic_vein_average_HU',
    'iliac_artery_left_volume_cm3',
    'iliac_artery_left_average_HU',
    'iliac_artery_right_volume_cm3',
    'iliac_artery_right_average_HU',
    'iliac_vena_left_volume_cm3',
    'iliac_vena_left_average_HU',
    'iliac_vena_right_volume_cm3',
    'iliac_vena_right_average_HU'
]
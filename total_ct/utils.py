import nibabel as nib
import numpy as np
from totalsegmentator.resampling import resample_img_cucim
# Utility functions - can be for random one offs that may need to be repeated

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


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   nnunet_resample=False, dtype=None, remove_negative=False, force_affine=None):
    """
    Resample nifti image to the new spacing (uses resample_img() internally).
    
    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results 
                     in a little bit less blurry results
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
    # This is a tweaked version of totalsegmentator's implementation
    data = img_in.dataobj[..., :]
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
        zoom = np.array(target_shape) / old_shape  
        new_spacing = img_spacing / zoom  
    else:
        zoom = img_spacing / new_spacing

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    # This is only correct if all off-diagonal elements are 0
    # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
    # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
    # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))


    new_data = resample_img_cucim(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # gpu resampling
        
    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    return nib.Nifti1Image(new_data, new_affine)
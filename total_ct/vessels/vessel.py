import cupy as cp
from totalsegmentator.map_to_binary import class_map

class Vessel:
    def __init__(self, orig_img: cp.ndarray, segmentation: cp.ndarray, spacing: list, task: str, vessel: str):
        """Class representing a vessel (such as an artery) within the body
        :param orig_img: the original image
        :param segmentation: the segmentation containing the vessel
        :param spacing: the pixel spacing of the image
        :param task: the task of totalsegmentator that generated the segmentation
        :param vessel: the name of the vessel (as it's called in totalsegmentator)
        """
        # We transpose the axes so axis 0 represents the z-slices
        self.orig = cp.flip(orig_img.transpose((2, 0, 1)), axis=(0, 2))
        
        # Only get the segmentation for the vessel we're interested in
        artery_val = {v: k for k, v in class_map[task].items()}.get(vessel)
        self.seg = (segmentation == artery_val).transpose((2, 0, 1)).astype(cp.int0)
        self.seg = cp.flip(self.seg, axis=(0, 2))
        # Set up the original mask
        self.hu_mask = cp.where(self.seg == 1, self.orig, 0)
        # Set up the pixel data
        y, x, z = spacing
        self.spacing = [z, y, x]
        self.voxel_size = z*y*x # This is in mm
        
        
    def measure_calcification(self, start: int=0, end: int=None, threshold: int=300) -> list:
        """Method to measure the calcification inside of a vessel segmentation
        :param start: The starting index to measure - default is the first slice of the image
        :param end: the ending index to stop measure - default is the final slice of the image
        :param threshold: the HU value to filter above for calcification. Default=300
        :return: a list holding the calcification in mm3 and the total region volume in mm3
        """
        # Region is the segmentation in HU, while volume is the total volume of the vessel region
        region = self.hu_mask[start:]
        volume = self.seg[start:] # This is binary 
        if end is not None:
            region = self.hu_mask[start:end]
            volume = self.seg[start:end]
        calc = region > threshold # Make the calcification a binary location
        total_calc = round(float(calc.sum() * self.voxel_size), 2)
        total_vol = round(float(volume.sum() * self.voxel_size), 2)
        return total_calc, total_vol
        
        
        
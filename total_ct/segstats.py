import os
import json
import nibabel as nib
import numpy as np
import pandas as pd
import cupy as cp
import concurrent.futures as concurrent_futures
import warnings

from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from types import SimpleNamespace
from tqdm import tqdm
from torch import cuda
from totalsegmentator.map_to_binary import class_map

from total_ct.utils import prefix_dict_keys, final_df_columns
from total_ct.vessels.cpr import measure_binary_diameter, CurvedPlanarReformation
from total_ct.vessels.thoracic_aorta import ThoracicAortaCPR

# Set using either gpu processes or numpy depending on if a GPU is available - won't perform all actions with this, but most
warnings.filterwarnings('ignore')
cuda_available = cuda.is_available()
if cuda_available:
    from cucim.skimage.measure import label, regionprops
else:
    from skimage.measure import label, regionprops
xp = cp if cuda_available else np
# Only measure a subset of items
total_map_indices = [*range(1, 25), *range(51, 69)]
total_map = {k: v for k, v in class_map['total'].items() if k < 69}
total_seg_map = {k: v for k, v in class_map['total'].items() if k in total_map_indices}

class SegStats:
    """A class to take a list of input CTs generate by CTBatch post-segmentation and perform analysis on their segmentations"""
    # We only run 3 of the 5 models, index cutoff is 69
    tasks = ['total', 'tissue_types', 'vertebrae_body', 'body']
    def __init__(self, input_cts, folders=False):
        """Initialize the base variables necessary to measure statistics
        :param input_cts: either a list of CTScan instances from CTBatch, or a list of segmentation folders
        :param folders: Boolean stating if the input_cts are a list of folders. Default=False
        """
        if xp == cp:
            print('GPU Detected. Running analysis primarily using cupy\n')
        # Perform a quick usability check - all the output segmentations must exist
        if folders:
            self.completed_cts, self.missing_segs = self._folder_list_check(input_cts)
        else:
            self.completed_cts, self.missing_segs = self._ct_list_check(input_cts)
        # Load in the base data necessary for each CT
        with open('missing_seg_folders.txt', 'a') as f:
            for folder in self.missing_segs:
                f.write(f'{folder}\n')
        # self._load_ct_variables()
    
    def process_segmentations(self, max_workers=4):
        """Run multiple threads in memory to speed up process"""
        with concurrent_futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.measure_segmentation, ct) for ct in self.completed_cts]
            for future in tqdm(
                    concurrent_futures.as_completed(futures),
                    desc='Measuring segmentations',
                    total=len(self.completed_cts),
                    dynamic_ncols=True):
                pass
        # # #     # executor.map(self.measure_segmentation, self.completed_cts)
        # for ct in tqdm(self.completed_cts, desc='Measuring Segmentations', dynamic_ncols=True):
        #     self.measure_segmentation(ct)
            
    def measure_segmentation(self, ct):
        """Method to analyze all of the segmentations from the output of total_segmentator and save the results"""
        try:
            self.load_ct_attributes(ct)
            self.measure_total_stats(ct)
            self.measure_vertebrae_body_and_diameter(ct)
            self.measure_tissue_stats(ct)
            self.measure_aorta_stats(ct)
            self.check_aorta(ct)
            self.write_output(ct)
            with open('completed_seg_folders.txt', 'a') as f:
                f.write(f'{ct.seg_folder}\t{ct.seg_folder}/{ct.nii_file_name}_stats.tsv\n')
        except Exception as e:
            print(e)
            with open('failed_seg_folders.txt', 'a') as f:
                f.write(f'{ct.seg_folder}\n')

        
    ######### Measurement Methods ########
    def measure_total_stats(self, ct):
        total_stats = {}
        vox_vol = ct.pixel_measurements['voxel_volume_cm3']
        for k, mask_name in total_seg_map.items():
            data = ct.total_seg == k
            total_stats[f'{mask_name}_volume_cm3'] = round(float(data.sum() * vox_vol), 2)
            roi_mask = (data > 0)
            avg_hu = xp.average(ct.orig_img, weights=roi_mask) if roi_mask.sum() > 0 else 0.0
            total_stats[f'{mask_name}_average_HU'] = round(float(avg_hu), 2)
        ct.total_stats = total_stats
        # return total_stats
        
    def measure_vertebrae_body_and_diameter(self, ct):
        cm_spacing = ct.pixel_measurements['pixel_spacing_cm']
        vert_bodies = xp.asanyarray(nib.load(f'{ct.seg_folder}/{ct.nii_file_name}_vertebrae_body.nii.gz').dataobj[..., :])
        verts = xp.where(vert_bodies == 1, ct.total_seg, 0)
        del vert_bodies
        vert_range = range(26, 51) # This is the range of indicies in the class map which correspond to vertebrae
        vert_vals = {}
        # Loop through the indices and measure the midline and diameter for each body
        for i in vert_range:
            name = total_map.get(i)
            arr = verts == i
            if arr.sum() != 0:
                z_indices = xp.where(arr)[-1]
                z_slices, frequencies = xp.unique(z_indices, return_counts=True)
                weighted_midline = int(xp.sum(z_slices * frequencies) / xp.sum(frequencies))
                average_axis, major_axis, minor_axis = measure_binary_diameter((arr[:, :, weighted_midline]), ct.pixel_measurements['y_x_spacing'])
                vert_vals[name] = {'midline': weighted_midline, 'average_axis_mm': average_axis, 'major_axis_diameter_mm': major_axis, 'minor_axis_diameter_mm': minor_axis}
            else:
                vert_vals[name] = {'midline': None, 'average_axis_mm': None, 'major_axis_diameter_mm': None, 'minor_axis_diameter_mm': None}
        ct.vertebrae_stats = vert_vals
            
    def measure_tissue_stats(self, ct):
        # These are the indices we want to measure at
        voi = [f'vertebrae_L{i}' for i in [1, 3, 5]]
        tissue_map = class_map['tissue_types']
        verts = {v.replace('vertebrae_', ''): ct.vertebrae_stats[v]['midline'] for v in voi if ct.vertebrae_stats[v]['midline'] is not None}
        tissue_stats = {}
        # Check if the values are available
        if len(verts) == 0:
            ct.tissue_stats = tissue_stats
            return 
        # Set the start and end point
        if len(verts) == 1:
            start = 0
        else:
            min_vert = min(verts, key=verts.get)
            start = verts[min_vert]
        max_vert = max(verts, key=verts.get)
        end = verts[max_vert] + 1
        # Get the necessary pixel_measurements
        vox_vol = ct.pixel_measurements['voxel_volume_cm3']
        pix_area = ct.pixel_measurements['pixel_area_cm2']
        pixel_spacing = ct.pixel_measurements['pixel_spacing_cm']
        # Separate the body from the segmentation
        body = xp.asanyarray(nib.load(f'{ct.seg_folder}/{ct.nii_file_name}_body.nii.gz').dataobj[..., :])
        body = body == 1
        # Load in the tissues
        tissues = xp.asanyarray(nib.load(f'{ct.seg_folder}/{ct.nii_file_name}_tissue_types.nii.gz').dataobj[..., :])
        body_abdominal = body[:, :, start:end]
        tissue_stats['total_body_volume'] = round(float(body.sum() * vox_vol), 2)
        tissue_stats['total_abdominal_body_volume'] = round(float(body_abdominal.sum() * vox_vol), 2)
        for idx, tissue in tissue_map.items():
            tissue_msk = tissues == idx
            tissue_stats[f'total_{tissue}_cm3'] = round(float((tissue_msk.sum() * vox_vol)), 2)
            tissue_stats[f'total_abdominal_{tissue}_cm3'] = round(float((tissue_msk[:, :, start:end].sum() * vox_vol)), 2)
        for vert, midline in verts.items():
            vert_body = self._largest_connected_component(body[:, :, midline])
            tissue_stats[f'{vert}_body_area_cm2'] = round(float((vert_body.area * pix_area)), 2)
            tissue_stats[f'{vert}_body_circ_cm'] = round(float(vert_body.perimeter * pixel_spacing), 2)
            tissue_stats[f'{vert}_body_circ_in'] = round(tissue_stats[f'{vert}_body_circ_cm'] / 2.54, 2)
            for idx, tissue in tissue_map.items():
                tissue_msk = tissues == idx
                vert_tissue = tissue_msk[:, :, midline]
                tissue_stats[f'{vert}_{tissue}_cm2'] = round(float((vert_tissue.sum() * pix_area)), 2)
        ct.tissue_stats = tissue_stats
        
    def measure_aorta_stats(self, ct, grid_size=80):
        aorta_stats = {}
        # if the aorta is not available - skip process
        if ct.total_stats['aorta_volume_cm3'] == 0:
            ct.aorta_stats = aorta_stats
            return
        # Need these measurements
        spacing = ct.spacing
        vox_vol = ct.pixel_measurements['voxel_volume']
        midlines = ct.vertebrae_stats
        # Calculate the number of z-slices that would make up 5cm
        n_slices_5_cm = int(50 / spacing[2])
        # Split the segmentation into into the abdomen and thorax
        # Abdomen
        abd_midlines = [midlines[f'vertebrae_L{i}']['midline'] for i in range(1, 6)]
        abd_midlines = [x for x in abd_midlines if x is not None]
        if abd_midlines:
            max_abd_midline = max(abd_midlines)
            # If the max_abd_midline is not at least 5 cm from the start of the CT, skip
            if max_abd_midline < n_slices_5_cm:
                max_abd_midline = None
        else:
            max_abd_midline = None
        # Abdominal split
        if max_abd_midline is not None:
            abd_orig = ct.orig_img[..., :max_abd_midline]
            abd_segs = ct.total_seg[..., :max_abd_midline]
            abd_aorta = CurvedPlanarReformation(abd_orig, abd_segs, spacing, 'total', 'aorta', 30)
            abd_measurements = abd_aorta.run_process(grid_size)
            abd_measurements = prefix_dict_keys(abd_measurements, 'abd_aorta')
            abd_calc, abd_vol = abd_aorta.measure_calcification()
            aorta_stats['abd_aorta_calc_mm3'] = abd_calc
            aorta_stats['abd_aorta_volume_mm3'] = abd_vol
            aorta_stats.update(abd_measurements)
        # Thorax
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
        if min_thor_midline is not None:
            thor_orig = ct.orig_img[..., min_thor_midline:]
            thor_segs = ct.total_seg[..., min_thor_midline:]
            thor_aorta = ThoracicAortaCPR(thor_orig, thor_segs, spacing, 'total', 'aorta', 35)
            asc_measurements, desc_measurements = thor_aorta.run_process(grid_size)
            asc_measurements = prefix_dict_keys(asc_measurements, 'asc_aorta')
            desc_measurements = prefix_dict_keys(desc_measurements, 'desc_aorta')
            thor_calc, thor_vol = thor_aorta.measure_calcification()
            aorta_stats['thor_aorta_calc_mm3'] = thor_calc
            aorta_stats['thor_aorta_volume_mm3'] = thor_vol
            aorta_stats.update(asc_measurements)
            aorta_stats.update(desc_measurements)
        ct.aorta_stats = aorta_stats
        
    def write_output(self, ct):
        final_stats = {}
        for data in [ct.header_info, ct.tissue_stats, ct.vertebrae_stats, ct.aorta_stats, ct.total_stats]:
            if data is ct.vertebrae_stats:
                vert_data = {}
                for k, v in data.items():
                    vert_data.update(prefix_dict_keys(v, k))
                final_stats.update(vert_data)
            else:
                final_stats.update(data)
        final_df = pd.DataFrame.from_dict(final_stats, orient='index').T
        final_df = final_df.reindex(columns=final_df_columns, fill_value=pd.NaT)
        final_df.to_csv(f'{ct.seg_folder}/{ct.nii_file_name}_stats.tsv', index=False, sep='\t')
        del ct.orig_ct
        del ct.orig_img
        del ct.total_seg
        del ct.total_stats
        del ct.tissue_stats
        del ct.vertebrae_stats
        del ct.aorta_stats
        cp.cuda.MemoryPool().free_all_blocks()
        
    def check_aorta(self, ct):
        if 'asc_aorta_avg_diam' in ct.aorta_stats:
            asc_diam = ct.aorta_stats.get('asc_aorta_avg_diam')
            if asc_diam is not None and asc_diam > 38:
                with open('TAA_asc_segmentations.txt', 'a') as f:
                    f.write(f'{ct.seg_folder}\t{asc_diam}\t{ct.aorta_stats.get("asc_aorta_major_diam")}\n')

        if 'desc_aorta_avg_diam' in ct.aorta_stats:
            desc_diam = ct.aorta_stats.get('desc_aorta_avg_diam')
            if desc_diam is not None and desc_diam > 38:
                with open('TAA_desc_segmentations.txt', 'a') as f:
                    f.write(f'{ct.seg_folder}\t{desc_diam}\t{ct.aorta_stats.get("desc_aorta_major_diam")}\n')
        if 'abd_aorta_avg_diam' in ct.aorta_stats:
            abd_diam = ct.aorta_stats.get('abd_aorta_avg_diam')
            if abd_diam is not None and abd_diam > 30:
                with open('AAA_segmentations.txt', 'a') as f:
                    f.write(f'{ct.seg_folder}\t{abd_diam}\t{ct.aorta_stats.get("abd_aorta_major_diam")}\n')
        
    ######### Utility Methods ########
    def load_ct_attributes(self, ct):
        ct.orig_ct = nib.load(ct.input)
        ct.orig_img = xp.asanyarray(ct.orig_ct.dataobj[..., :].astype(np.float32))
        # ct.orig_img = xp.asanyarray(ct.orig_ct.get_fdata(caching='unchanged'))
        total_file = f'{ct.seg_folder}/{ct.nii_file_name}_total.nii.gz'
        # If the fast model was run- change the filename
        if not os.path.exists(total_file):
            total_file = f'{ct.seg_folder}/{ct.nii_file_name}_total_fast.nii.gz'
        total_seg = nib.load(total_file)
        ct.total_seg = xp.asanyarray(total_seg.dataobj[..., :])
        # ct.total_seg = xp.asanyarray(total_seg.get_fdata(caching='unchanged'))
        ct.spacing = total_seg.header.get_zooms()
        ct.pixel_measurements = self._create_pixel_measurements(ct)
    
    def _largest_connected_component(self, arr):
        arr = xp.asarray(arr)
        labeled_arr = label(arr)
        regions = regionprops(labeled_arr)
        lcc = max(regions, key=lambda x: x.area)
        return lcc
    
    @classmethod
    def _folder_list_check(cls, folders):
        """Set up a list of CT instances using SimpleNameSpace to be comparable to a list of CTScans"""
        analyzable_cts = []
        missing_segs = []
        for folder in folders:
            # Make sure folder has the proper formatting
            if folder[-1] == '/':
                folder = folder[:-1]
            # Define the holding folder
            series_folder = '/'.join(folder.split('/')[:-1])
            # Search for TS output files
            seg_files = glob(f'{folder}/*.nii.gz')
            if len(seg_files) == 0:
                print(f'{folder} is empty!')
                missing_segs.append(folder)
                continue
            # Check that the nifti has all the requisite segmentations
            has_segs = []
            for seg in ['total', 'body', 'tissue_types', 'vertebrae']:
                if any([seg in output for output in seg_files]):
                    if seg == 'body': 
                        if any([seg in output and 'vertebrae' not in output for output in seg_files]):
                            has_segs.append(True)
                        else:
                            has_segs.append(False)
                    else:
                        has_segs.append(True)
                else:
                    has_segs.append(False)
            if not all(has_segs):
                print(f'{folder} does not have all necessary segmentations!')
                missing_segs.append(folder)
                continue
            # Find the base name of the nifti file
            for output in seg_files:
                if 'tissue_types' in output:
                    tissue_file = output
            nii_file_name = tissue_file.split('/')[-1].replace('_tissue_types.nii.gz', '')
            # Check that the original CT and it's header file exist
            if not os.path.exists(f'{series_folder}/{nii_file_name}.nii.gz'):
                print(f'{folder} does not have the original CT file!')
                missing_segs.append(folder)
                continue
            if not os.path.exists(f'{series_folder}/header_info.json'):
                print(f"{folder} does not have it's header data stored as a json!")
                missing_segs.append(folder)
                continue
            # Set up namespace for CT that would match an input from the Segmentator class
            ct = SimpleNamespace()
            ct.input = f'{series_folder}/{nii_file_name}.nii.gz'
            ct.nii_file_name = nii_file_name
            ct.seg_folder = folder
            with open(f'{series_folder}/header_info.json') as f:
                ct.header_info = json.load(f)
            ct.mrn = ct.header_info['mrn']
            ct.accession = ct.header_info['accession']
            ct.series_name = ct.header_info['series_name']
            if ct.header_info.get('spacing') is None:
                ct.header_info['spacing'] = [ct.header_info['length_mm'], ct.header_info['width_mm'], ct.header_info['slice_thickness_mm']]
            ct.spacing = ct.header_info['spacing']
            analyzable_cts.append(ct)
        return analyzable_cts, missing_segs
            
                
    @classmethod
    def _ct_list_check(cls, input_cts):
        """a method to check if the input_cts are usable, and if their appropriate output segmentations exist
        :param input_cts: the input to the class initialization
        :return: the list of CTScan instances to analyze
        """
        analyzable_cts = []
        missing_segs = []
        for input_ct in tqdm(input_cts, desc='Checking that CTs have all segmentations', dynamic_ncols=True):
            if input_ct.usable:
                for i, nii in enumerate(input_ct.input):
                    ct = deepcopy(input_ct)
                    nifti_name = ct.nii_file_name[i]
                    output_dir = f'{ct.output_dir}/{nifti_name}_segs'
                    if os.path.exists(output_dir):
                        seg_exists = []
                        for task in cls.tasks:
                            seg_completed = os.path.exists(f'{output_dir}/{nifti_name}_{task}.nii.gz')
                            if seg_completed:
                                seg_exists.append(True)
                            elif task == 'total' and not seg_completed:
                                if os.path.exists(f'{output_dir}/{nifti_name}_{task}_fast.nii.gz'):
                                    seg_exists.append(True)
                                else:
                                    seg_exists.append(False)
                            else:
                                seg_exists.append(False)
                        if all(seg_exists):
                            ct.input = nii
                            ct.nii_file_name = nifti_name
                            ct.seg_folder = output_dir
                            analyzable_cts.append(ct)
                        else:
                            print(f'{nii} does not have all segmentations!\n')
                            missing_segs.append(output_dir)
                    else:
                        print(f'{output_dir} does not exist!')
                        missing_segs.append(output_dir)
        return analyzable_cts, missing_segs
    
    
    @classmethod
    def _create_pixel_measurements(cls, ct):
        l, w, h = ct.spacing
        pixel_vals = {
            'length': l,
            'width': w,
            'height': h,
            'y_x_spacing': (l, w),
            'voxel_volume': l * w * h,
            'voxel_volume_cm3': l * w * h / 1000,
            'pixel_area': l * w,
            'pixel_area_cm2': l * w / 100,
            'pixel_spacing': (l + w) / 2,
            'pixel_spacing_cm': ((l + w) / 2) / 10
        }
        return pixel_vals

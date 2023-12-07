import os
import time
import torch
import nibabel as nib
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
from shutil import rmtree, move

from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from totalsegmentator.libs import check_if_shape_and_affine_identical, nostdout
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname
from totalsegmentator.resampling import change_spacing
from totalsegmentator.postprocessing import remove_auxiliary_labels, keep_largest_blob_multilabel, remove_small_blobs_multilabel
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from total_ct.ctbatch import CTBatch


class TotalSeg(CTBatch):
    """Child class of CTBatch, used to run all the totalsegmentator models on list of CTs"""
    tasks = ['total', 'tissue_types', 'vertebrae_body', 'body']
    def segment_batch(self):
        self.last_resample = None
        self.model_configs = self.create_task_configs(self.args.fast)
        self.set_up_batch() # This is from the CTBatch class
        self.get_batch_data()
        self.make_predictor()
        for task in self.tasks:
            self.preprocess_batch(task)
            self.predict_batch(task)
            self.postprocess_batch(task)
    
    def get_batch_data(self):
        "extract usable ct files meta-data and spacing"
        ct_dict = {}
        print('Gathering batch data...')
        for ct in self.input_cts:
            if ct.usable:
                # If input was a dicom folder, there could be > 1 nifti file created
                for i, nii in enumerate(ct.input):
                    image = nib.load(nii)
                    img_in = nib.Nifti1Image(image.get_fdata(), image.affine)
                    img_in = as_closest_canonical(img_in)
                    ct_dict[nii] = {
                        'header': image.header,
                        'img_in_orig': image,
                        'img_in': img_in,
                        'spacing': img_in.header.get_zooms(),
                        'shape': img_in.shape,
                        'affine': img_in.affine,
                        'mrn': ct.mrn,
                        'accession': ct.accession,
                        'series_name': ct.series_name,
                        'nii_file_name': ct.nii_file_name[i],
                        'output_dir': f'{ct.output_dir}/{ct.nii_file_name[i]}_segs',
                        'tmp_name': f'{ct.mrn}_{ct.accession}_{ct.series_name}_{ct.nii_file_name[i]}'
                    }
        self.batch_data = ct_dict
        
    def make_predictor(self):
        "Method to set up the nnUnet predictor that will load various weights"
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            device = torch.device('cuda')
        else:
            raise RuntimeError('No GPU detected. TotalSegmentator will not run')
        
        step_size = 0.5
        disable_tta = True
        self.predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_gpu=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        
    ##### Preprocess for TotalSegmentator #####
    def preprocess_batch(self, task):
        print(f'Preprocessing for {task} task...')
        task_config = self.model_configs[task]
        resample_size = task_config['resample']
        # This is the maximum that a ct can be in size without splitting
        nr_voxels_thr = 256*256*900
        # Track the tasks image resampling, if task spacing is repeated, we don't have to preprocess again
        if self.last_resample is None:
            self.last_resample = resample_size    
        if (resample_size == self.last_resample) and task != 'total':
            print('Resampled array size is the same as previous task, reusing files.\n')
            return
        else:
            self._make_tmp_dirs()
        # Begin preprocessing images to match task-specific inputs
        for nifti, nifti_dict in tqdm(self.batch_data.items()):
            img_in = nifti_dict['img_in']
            img_in_rsp = change_spacing(img_in, [resample_size, resample_size, resample_size], order=3, dtype=np.int32, nr_cpus=6)
            rsp_shape = img_in_rsp.shape
            self.batch_data[nifti]['resampled_shape'] = rsp_shape
            self.batch_data[nifti]['rsp_affine'] = img_in_rsp.affine
            do_triple_split = np.prod(rsp_shape) > nr_voxels_thr and rsp_shape[2] > 200
            self.batch_data[nifti]['triple_split'] = do_triple_split
            if do_triple_split:
                print(f'splitting {nifti} into 3 parts...')
                third = rsp_shape[2] // 3
                margin = 20
                img_in_rsp_data = img_in_rsp.get_fdata()
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
                    f'{self.tmp_in}/{nifti_dict["tmp_name"]}_s01_0000.nii.gz')
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
                    f'{self.tmp_in}/{nifti_dict["tmp_name"]}_s02_0000.nii.gz')
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
                    f'{self.tmp_in}/{nifti_dict["tmp_name"]}_s03_0000.nii.gz')
            else:
                nib.save(img_in_rsp, f'{self.tmp_in}/{nifti_dict["tmp_name"]}_0000.nii.gz')
                
        self.last_resample = resample_size
        print('Done Preprocessing!\n')
        
    ### Predict with Total Segmentator ###
    def predict_batch(self, task):
        plans = "nnUNetPlans"
        task_config = self.model_configs[task]
        task_id = task_config['task_id']
        trainer = task_config['trainer']
        model = task_config['model']
        folds = task_config['folds']
        chk = "checkpoint_final.pth"
        verbose = False
        save_probabilities = False
        continue_prediction = False
        npp = 2
        nps = 6
        prev_stage_predictions = None
        num_parts = 1
        part_id = 0
        print(f'Predicting {task} model...')
        st = time.time()
        if not isinstance(task_id, list):
            with nostdout(False):
                print(f'Task Configurations: {task_config}')
                model_folder = get_output_folder(task_id, trainer, plans, model)
                self.predictor.initialize_from_trained_model_folder(
                    model_folder,
                    use_folds=folds,
                    checkpoint_name=chk
                )
                self.predictor.predict_from_files(
                    self.tmp_in,
                    self.tmp_out,
                    save_probabilities=save_probabilities,
                    overwrite=not continue_prediction,
                    num_processes_preprocessing=npp,
                    num_processes_segmentation_export=nps,
                    folder_with_segs_from_prev_stage=prev_stage_predictions,
                    num_parts=num_parts,
                    part_id=part_id
                )
        else:
            # Need to implement logic for High resolution Total Task
            self.multi_task_id_predict(task_id, task, trainer, model)
        print(f'Batch Done. Predicted in {time.time() - st:.2f}!\n')
        
    def multi_task_id_predict(self, task_id_list, task, trainer, model):
        plans='nnUNetPlans'
        chk = 'checkpoint_final.pth'
        verbose = False
        save_probabilities = False
        continue_prediction = False
        npp = 2
        nps = 6
        prev_stage_predictions = None
        num_parts = 1
        part_id = 0
        all_names = []
        # Get a list of all the eventual output names
        for ct, nifti_dict in self.batch_data.items():
            if nifti_dict['triple_split']:
                ct_names = [f'{self.tmp_out}/{nifti_dict["tmp_name"]}_{s}.nii.gz' for s in ['s01', 's02', 's03']]
                for name in ct_names:
                    all_names.append(name)
            else:
                all_names.append(f'{self.tmp_out}/{nifti_dict["tmp_name"]}.nii.gz')
                
        for i, task_id in enumerate(task_id_list):
            print(f'Predicting part {i+1} of {len(task_id_list)}...')
            with nostdout(False):
                st = time.time()
                model_folder = get_output_folder(task_id, trainer, plans, model)
                self.predictor.initialize_from_trained_model_folder(
                    model_folder,
                    use_folds=[0],
                    checkpoint_name=chk
                )
                self.predictor.predict_from_files(
                    self.tmp_in,
                    self.tmp_out,
                    save_probabilities=save_probabilities,
                    overwrite=not continue_prediction,
                    num_processes_preprocessing=npp,
                    num_processes_segmentation_export=nps,
                    folder_with_segs_from_prev_stage=prev_stage_predictions,
                    num_parts=num_parts,
                    part_id=part_id
                )
            print(f'Batch part {i+1} done. Predicted in {time.time() - st:.2f}!\n')
            # Move the outputs to task-specific names
            for ct_name in all_names:
                name = ct_name.replace('.nii.gz', '')
                new_name = f'{name}_{task_id}.nii.gz'
                move(ct_name, new_name)
        # No multiprocessing
        self._combine_task_outputs(task_id_list, task)
        # Multiprocessing
        # combine_multi_mask_mp(self.batch_data, task_id_list, task, self.tmp_out)
         
    
    def _combine_task_outputs(self, task_id_list, task):
        class_map_inv = {v: k for k, v in class_map[task].items()}
        seg_combined = {}
        for ct, nifti_dict in self.batch_data.items():
            if nifti_dict['triple_split']:
                img_parts = [f'{nifti_dict["tmp_name"]}_{s}' for s in ['s01', 's02', 's03']]
            else:
                img_parts = [f'{nifti_dict["tmp_name"]}']
            for img_part in img_parts:
                # Get the original shape
                img_shape = nifti_dict['resampled_shape']
                seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                
                for task_id in task_id_list:
                    seg = nib.load(f'{self.tmp_out}/{img_part}_{task_id}.nii.gz').get_fdata()
                    for jdx, class_name in class_map_5_parts[map_taskid_to_partname[task_id]].items():
                        seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
            
                nib.save(nib.Nifti1Image(seg_combined[img_part], nifti_dict['rsp_affine']), f'{self.tmp_out}/{img_part}.nii.gz')
                
    #### Postprocessing #######
    def postprocess_batch(self, task):
        """
        Reshape batch outputs to original spacing and save them as nifti files to their appropriate directory
        """
        resample = self.last_resample
        print(f'Postprocessing {task} results...')
        for ct, ct_info in tqdm(self.batch_data.items()):
            # ct_info = self.batch_info[ct]
            if ct_info['triple_split']:
                third = ct_info['resampled_shape'][-1] // 3
                margin = 20
                combined_img = np.zeros(ct_info['resampled_shape'], dtype=np.uint8)
                combined_img[:,:,:third] = nib.load(f'{self.tmp_out}/{ct_info["tmp_name"]}_s01.nii.gz').get_fdata()[:,:,:-margin]
                combined_img[:,:,third:third*2] = nib.load(f'{self.tmp_out}/{ct_info["tmp_name"]}_s02.nii.gz').get_fdata()[:,:,margin-1:-margin]
                combined_img[:,:,third*2:] = nib.load(f'{self.tmp_out}/{ct_info["tmp_name"]}_s03.nii.gz').get_fdata()[:,:,margin-1:]
                nib.save(nib.Nifti1Image(combined_img, ct_info['rsp_affine']), f"{self.tmp_out}/{ct_info['tmp_name']}.nii.gz")
            img_in_orig = ct_info['img_in_orig']
            os.makedirs(ct_info['output_dir'], exist_ok=True)
            img_pred = nib.load(f'{self.tmp_out}/{ct_info["tmp_name"]}.nii.gz')
            img_pred = remove_auxiliary_labels(img_pred, task)
            # Body Specific transformations
            with nostdout(False):
                if task == 'body':
                    img_pred_pp = keep_largest_blob_multilabel(
                        img_pred.get_fdata().astype(np.uint8),
                        class_map[task],
                        ["body_trunc"]
                    )
                    img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)
                    vox_vol = np.prod(img_pred.header.get_zooms())
                    size_thr_mm3 = 50000 / vox_vol
                    img_pred_pp = remove_small_blobs_multilabel(
                        img_pred.get_fdata().astype(np.uint8),
                        class_map[task],
                        ["body_extremities"],
                        interval=[size_thr_mm3, 1e10]
                    )
                    img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

                img_pred = change_spacing(
                    img_pred,
                    [resample, resample, resample],
                    ct_info['shape'],
                    order=0,
                    dtype=np.uint8,
                    nr_cpus=4,
                    force_affine=ct_info['affine']
                )
                img_pred = undo_canonical(img_pred, img_in_orig)
                check_if_shape_and_affine_identical(img_in_orig, img_pred)
                img_data = img_pred.get_fdata().astype(np.uint8)
                new_header = ct_info['header']
                new_header.set_data_dtype(np.uint8)
                img_out = nib.Nifti1Image(img_data, affine=img_pred.affine, header=new_header)
                if self.args.fast and task == 'total':
                    task_file = f"{ct_info['output_dir']}/{ct_info['nii_file_name']}_{task}_fast.nii.gz"
                    # nib.save(img_out, f"{ct_info['output_dir']}/{ct_info['nii_file_name']}_{task}_fast.nii.gz")
                else:
                    task_file = f"{ct_info['output_dir']}/{ct_info['nii_file_name']}_{task}.nii.gz"
                nib.save(img_out, task_file)
                self.batch_data[ct][task] = task_file 
        rmtree(self.tmp_out)
        os.makedirs(self.tmp_out)
        print('Done!\n')
        
    ### Utility functions ###
    def _make_tmp_dirs(self):
        # Not sure yet if I want to set every CTScan instance to have tmp_in/tmp_out attributes
        tmp_in = f'{self.args.tmp_dir}/nnunet_pre'
        tmp_out = f'{self.args.tmp_dir}/nnunet_out'
        for path in [tmp_in, tmp_out]:
            if os.path.exists(path):
                rmtree(path)
            os.makedirs(path)
        self.tmp_in = tmp_in
        self.tmp_out = tmp_out
    
    
    @staticmethod
    def create_task_configs(fast):
        crop_addon = [3, 3, 3] # Default value
        config = {
            'total': {},
            'body': {},
            'vertebrae_body': {},
            'tissue_types': {},
            'coronary_arteries': {}
        }
        for task in config.keys():
            # Full Segmentation
            if task == 'total':
                if fast:
                    config['total']['task_id'] = 297
                    config['total']['resample'] = 3.0
                    config['total']['trainer'] = 'nnUNetTrainer_4000epochs_NoMirroring'
                    config['total']['crop'] = None
                else:
                    config['total']['task_id'] = [291, 292, 293]
                    config['total']['resample'] = 1.5
                    config['total']['trainer'] = 'nnUNetTrainerNoMirroring'
                    config['total']['crop'] = None
                config['total']['model'] = '3d_fullres'
                config['total']['folds'] = [0]
            # Body Segmentation
            elif task == "body":
                config['body']['task_id'] = 299
                config['body']['resample'] = 1.5
                config['body']['trainer'] = "nnUNetTrainer"
                config['body']['crop'] = None
                config['body']['model'] = "3d_fullres"
                config['body']['folds'] = [0]
            # Vertebrae Body Segmentation
            elif task == "vertebrae_body":
                config['vertebrae_body']['task_id'] = 302
                config['vertebrae_body']['resample'] = 1.5
                config['vertebrae_body']['trainer'] = "nnUNetTrainer"
                config['vertebrae_body']['crop'] = None
                config['vertebrae_body']['model'] = "3d_fullres"
                config['vertebrae_body']['folds'] = [0] 
            # Tissue Types Segmentation  
            elif task == "tissue_types":
                config['tissue_types']['task_id'] = 481
                config['tissue_types']['resample'] = 1.5
                config['tissue_types']['trainer'] = "nnUNetTrainer"
                config['tissue_types']['crop'] = None
                config['tissue_types']['model'] = "3d_fullres"
                config['tissue_types']['folds'] = [0]
            # Coronary Arteries Segmentation
            elif task == "coronary_arteries":
                config['coronary_arteries']['task_id'] = 503
                config['coronary_arteries']['resample'] = None
                config['coronary_arteries']['trainer'] = "nnUNetTrainer"
                config['coronary_arteries']['crop'] = ["heart"]
                config['coronary_arteries']['model'] = "3d_fullres"
                config['coronary_arteries']['folds'] = [0]
        return config
    

def merge_multi_task_masks(nifti_dict, task_id_list, task, tmp_out):
    class_map_inv = {v: k for k, v in class_map[task].items()}
    seg_combined = {}
    if nifti_dict['triple_split']:
        img_parts = [f'{nifti_dict["tmp_name"]}_{s}' for s in ['s01', 's02', 's03']]
    else:
        img_parts = [f'{nifti_dict["tmp_name"]}']
    for img_part in img_parts:
        # Get the original shape
        img_shape = nifti_dict['resampled_shape']
        seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
        for task_id in task_id_list:
            seg = nib.load(f'{tmp_out}/{img_part}_{task_id}.nii.gz').get_fdata()
            for jdx, class_name in class_map_5_parts[map_taskid_to_partname[task_id]].items():
                seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
    nib.save(nib.Nifti1Image(seg_combined[img_part], nifti_dict['rsp_affine']), f'{tmp_out}/{img_part}.nii.gz')


def combine_multi_mask_mp(batch_data, task_id_list, task, tmp_out):
    with Pool(processes=4) as pool:
        items = [(nifti_dict, task_id_list, task, tmp_out) for _, nifti_dict in batch_data.items()]
        pool.starmap(merge_multi_task_masks, items)

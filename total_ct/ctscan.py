import os
import json
import natsort
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
import numpy as np
from datetime import datetime
from glob import glob
from shutil import copy


class CTScan:
    """
    Class holding the metadata/file info for a CT scan. Can be either a Dicom folder or Nifti file. Will convert Dicom to nifti automatically.
    """
    orientations = {
        'COR': [1, 0, 0, 0, 0, -1],
        'SAG': [0, 1, 0, 0, 0, -1],
        'AX': [1, 0, 0, 0, 1, 0]
    }
    important_attributes = {
        'PatientID': 'mrn',
        'AccessionNumber': 'accession',
        'SeriesDescription': 'series_name',
        'ImageOrientationPatient': 'ct_direction',
        'ImageType': 'image_type',
        'PatientSex': 'sex',
        'PatientBirthDate': 'birthday',
        'AcquisitionDate': 'scan_date',
        'PatientAge': 'age_at_scan',
        'PixelSpacing': 'pixel_spacing',
        'SliceThickness': 'slice_thickness_mm',
        'Manufacturer': 'manufacturer',
        'ManufacturerModelName': 'manufacturer_model',
        'KVP': 'kvp',
        'SequenceName': 'sequence_name',
        'ProtocolName': 'protocol_name',
        'ContrastBolusAgent': 'contrast_bolus_agent',
        'ContrastBolusRoute': 'contrast_bolus_route',
        'MultienergyCTAcquisitionSequence': 'multienergy_ct'
    }

    def __init__(self, input_ct, output_dir, save_copy):
        """
        :param input_ct: the path to nifti file or dicom folder
        :param output_dir: the base directory to store all the results in subfolders.
        """
        # This is the base output directory, will save to mrn/acc/series_name
        self.input_ct = input_ct
        self.base_dir = output_dir
        # Set the input to be a NifTi file, convert a dicom folder if necessary
        if os.path.isfile(input_ct) and ('.nii.gz' in input_ct or '.nii' in input_ct):
            self.set_nifti_attr(input_ct, save_copy)
            
        elif os.path.isdir(input_ct):
            # This will also set self.usable to True or False based on what happens with the conversion
            # will also set self.nii_file_name
            self.convert_dir_to_nifti(input_ct)
        else:
            # if the input is not a nifti file or dicom folder, set self.usable to be false
            print(f'Input {input_ct} is not a Nifti file or Dicom folder, will be skipped in analysis.\n')
            self.usable = False
            
    def set_nifti_attr(self, input_ct, save_copy):
        """method to set the necessary attributes for a nifti file"""
        nifti_folder = '/'.join(input_ct.split('/')[:-1])
        self.input = [input_ct]
        self.nii_file_name = [x.split('/')[-1].replace('.nii.gz', '') for x in self.input]
        
        if os.path.exists(f'{nifti_folder}/header_info.json'):
            self.set_header_attr(f'{nifti_folder}/header_info.json')
            self.usable = True
        else:
            print(f'{input_ct} has no header file, it will be skipped in analysis\n')
            self.usable = False
        # Set the output directory 
        self.output_dir = f'{self.base_dir}/{self.mrn}/{self.accession}/{self.series_name}/'
        if not os.path.exists(self.output_dir):
            print(f'Output path does not exist, generating {self.output_dir}\n')
            os.makedirs(self.output_dir)
        if save_copy:
            copy(input_ct, self.output_dir)
            
    def set_header_attr(self, header_file=None):
        if header_file is not None:
            self.header_file = header_file
            with open(header_file, 'r') as f:
                self.header_info = json.load(f)
        self.mrn = self.header_info['mrn']
        self.accession = self.header_info['accession']
        self.series_name = self.header_info['series_name']
        if self.header_info.get('spacing') is None:
            self.header_info['spacing'] = [self.header_info['length_mm'], self.header_info['width_mm'], self.header_info['slice_thickness_mm']]
        self.spacing = self.header_info['spacing']
        
    def load_dicom_header(self, input_folder):
        """
        Method that loads the dicom header and sets self.header_data to be a dictionary containing the info \n
        :param header: the dicom header object
        """
        dicom_file = glob(f'{input_folder}/*')[0]
        header = pydicom.dcmread(dicom_file)
        series_info = {}
        # Loop through all the important attributes
        for tag, column in self.important_attributes.items():
            try:
                value = getattr(header, tag)
                if tag == "SeriesDescription":
                    value = value.replace('/', '_').replace(' ', '_').replace('-', '_').replace(
                        ',', '.')
                elif tag == "ImageOrientationPatient":
                    orientation = np.round(getattr(header, tag))
                    value = None
                    for key, direction in self.orientations.items():
                        if np.array_equal(orientation, direction):
                            value = key
                            break
                elif tag == 'ImageType':
                    value = '_'.join(value)
                elif tag == 'PatientBirthDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'AcquisitionDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'PatientAge':
                    value = int(value[:-1])
                elif tag == 'PixelSpacing':
                    # Value in this case is a list
                    length, width = value
                    series_info['length_mm'] = length
                    series_info['width_mm'] = width
                    continue
                elif tag == 'PatientSex':
                    if value == 'M':
                        value = 0
                    elif value == 'F':
                        value = 1
                elif tag == 'ContrastBolusAgent' or tag == 'ContrastBolusRoute':
                    value = 1
                elif tag == 'MultienergyCTAcquisitionSequence':
                    value = 1

                series_info[column] = value

            except AttributeError:
                series_info[column] = None
        series_info['spacing'] = [float(header.PixelSpacing[0]), float(header.PixelSpacing[1]), float(header.SliceThickness)]
        self.header_info = series_info
        self.set_header_attr()
        self.first_dicom = dicom_file
        # copy(dicom_file, f'{self.base_dir}/{self.mrn}/{self.accession}/{self.series_name}')
    
    def convert_dir_to_nifti(self, input_folder):
        num_files = len(glob(f'{input_folder}/*'))
        if num_files < 20:
            print(f'Conversion will not occur, less than 20 dicom files in folder {input_folder}.\n')
            self.usable = False
            return
        self.load_dicom_header(input_folder)
        self.output_dir = f'{self.base_dir}/{self.mrn}/{self.accession}/{self.series_name}'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        try:
            copy(self.first_dicom, self.output_dir)
            print(f'Converting {input_folder} to Nifti Format...\n')    
            settings.enable_resampling = True
            settings.resampling_order = 1
            settings.reorient_nifti = False
            dicom2nifti.convert_directory(input_folder, self.output_dir, compression=True, reorient=True)
            self.header_file = f'{self.output_dir}/header_info.json'
            with open(self.header_file, 'w') as f:
                json.dump(self.header_info, f, indent=4)
            print(f'Series successfully converted! Saved to {self.output_dir}.\n')
            self.usable = True
            nifti_files = glob(f'{self.output_dir}/*.nii.gz')
            self.nii_file_name = [x.split('/')[-1].replace('.nii.gz', '') for x in nifti_files]
            self.input = nifti_files
        
        except Exception as e:
            print(f"Error converting DICOM to NIfTI: {e} - will be skipped in analysis\n")
            self.usable = False
            return None

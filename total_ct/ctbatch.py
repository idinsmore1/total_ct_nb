import os
from functools import partial
from multiprocessing import Pool
from shutil import rmtree
from types import SimpleNamespace
from tqdm import tqdm
from total_ct.ctscan import CTScan


class CTBatch:
    """Class representing a batch of CT inputs - either dicom folders or numpy files.
    Handles the logic for converting to nifti, as well as pre- and post- processing.
    """
    def __init__(self, batch_list, args, num_workers=8):
        """
        :param batch_list: the list of ct scans to be analyzed
        :param args: the input arguments to main.py
        :param num_workers: The number of multiprocessing workers to use. Default = 4.
        """
        self.batch_list = batch_list
        self.args = args
        self.num_workers = num_workers
        
    def set_up_batch(self):
        # self.input_cts is the list of CTScan instances
        self.input_cts = instantiate_batch(self.batch_list, self.args, self.num_workers)


def instantiate_ct(input_ct, output_dir, save_copy) -> CTScan:
    """The worker function for multiprocessing through a list of CTs
    :param input_ct: the path to the input ct nifti file or dicom folder
    :param output_dir: the output directory to save the results to
    :return: returns a CTScan instance
    """
    ctscan = CTScan(input_ct, output_dir, save_copy)
    return ctscan
    

def instantiate_batch(batch_list, args, num_workers):
    """Multiprocessing function for setting up CT class instances (converts dicom to Nifti as well)
    :param batch_list: the list of input files/folders
    :param args: the input arguments to main.py
    :return: A list of CTScan Instances (necessary for keeping track of filepaths)
    """
    # input_cts = []
    # for ct in tqdm(batch_list):
    #     input_cts.append(instantiate_ct(ct, args.output_dir, args.save_copy))
    # return input_cts
    # self.input_cts = input_cts
    partial_instantiate = partial(instantiate_ct, output_dir=args.output_dir, save_copy=args.save_copy)
    with Pool(processes=num_workers) as pool:
        tasks = [(i, args.output_dir, args.save_copy) for i in batch_list]
        input_cts = pool.starmap(instantiate_ct, tasks)
        # input_cts = pool.imap(partial_instantiate, batch_list)
    return input_cts
            
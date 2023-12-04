# This is the main tool of total-ct, segments and analyzes batches of CTs automatically.
# There will also be a analyze.py script, which only performs the analysis section on
# segmented cts.
import os
import argparse

from itertools import islice
from math import ceil
from totalsegmentator.config import setup_nnunet
setup_nnunet()
from total_ct.analyze import CTStats
from total_ct.inference import TotalSeg
from total_ct.utils import block_print


parser = argparse.ArgumentParser('Total-CT Batch Edition. Leverages TotalSegmentator models to run analysis on multiple CTs within the Geisinger Medical System.')
parser.add_argument('--input-files', '-i', type=str, help='the input list of ct scans in either Dicom of Nifti format (.txt file)')
parser.add_argument('--output-dir', '-o', type=str, help='The directory to store the segmentations and metrics in (will be in a subfolder based on ct metadata')
parser.add_argument('--gpu', '-g', type=str, default='0', help='The gpu index set by nvidia-smi to be used to run the analysis. Default=0')
parser.add_argument('--batch-size', '-b', type=int, default=10, help='The batch size for each run of nnUNet to process. Default=10')
parser.add_argument('--tmp-dir', '-t', type=str, help='The temporary directory to store preprocessed and predicted outputs')
parser.add_argument('--fast', '-f', action='store_true', help='Flag to use the faster 3mm model for the "total" segmentation')
parser.add_argument('--total-only', '-to', action='store_true', help='Flag to only run the "total" segmentation - biggest and most important')
parser.add_argument('--save-copy', '-s', action='store_true', help='Flag to save a copy of the input CT to the output folder (this will automatically occur if the input ct is a dicom folder')
parser.add_argument('--tracker-file', '-tf', type=str, default=None, help='Set the specific file to track finished folders to. If not set, will save to ./gpu{-g argument}_completed.txt')
args = parser.parse_args()

# Set the global environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



def main():
    # Set up tracker file
    if args.tracker_file is None:
        args.tracker_file = f'gpu{args.gpu}_completed.txt'
    # Remove the final forward-slash if it exists for easier reading
    if args.output_dir[-1] == '/':
        args.output_dir = args.output_dir[:-1]
    # Read in the input list
    with open(args.input_files) as f:
        all_ct_files = f.read().splitlines()
        ct_iterator = iter(all_ct_files)
    # Give a quick overview
    total_batches = ceil(len(all_ct_files) / args.batch_size)
    block_print(
        f'Total Number of CT Scans: {len(all_ct_files)}',
        f'Batch Size: {args.batch_size}',
        f'Total Number of Batches: {total_batches}'
    )
    # Begin Analyzation Loop
    batch_num = 0
    while True:
        if batch_num > total_batches:
            print('All CTs analyzed!! Quitting.')
            break
        block_print(f'Predicting Batch {batch_num} / {total_batches}...')
        # Get a list of inputs batch_size long
        batch_list = list(islice(ct_iterator, args.batch_size))
        # Pass the batch list to the TotalSeg class - will handle meta-data
        batch = TotalSeg(batch_list, args)
        batch.segment_batch()
        finished_cts = batch.input_cts
        # Delete the batch instance to save memory
        del batch
        # Pass the finished cts to the CTStats class 
        stats = CTStats(finished_cts)
        
        batch_num += 1
        
    
if __name__ == '__main__':
    main()
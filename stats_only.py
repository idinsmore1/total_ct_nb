import os
import argparse

from itertools import islice
from math import ceil
from totalsegmentator.config import setup_nnunet
setup_nnunet()

parser = argparse.ArgumentParser('Total-CT Batch Edition. Leverages TotalSegmentator models to run analysis on multiple CTs within the Geisinger Medical System.')
parser.add_argument('--input-folders', '-i', type=str, help='the input list of completed ct scans segmentations')
parser.add_argument('--gpu', '-g', type=str, default='0', help='The gpu index set by nvidia-smi to be used to run the analysis. Default=0')
parser.add_argument('--batch-size', '-b', type=int, default=10, help='The batch size for each run of nnUNet to process. Default=10')
args = parser.parse_args()

# set global environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from total_ct.segstats import SegStats
from total_ct.utils import block_print

def main():
    # load the folders
    with open(args.input_folders) as f:
        all_folders = f.read().splitlines()
        ct_iterator = iter(all_folders)
        
    total_batches = ceil(len(all_folders) / args.batch_size)
    block_print(
        f'Total Number of CT Scans: {len(all_folders)}',
        f'Batch Size: {args.batch_size}',
        f'Total Number of Batches: {total_batches}'
    )
    # Begin loop
    batch_num = 1
    while True:
        if batch_num > total_batches:
            print('All CTs analyzed!! Quitting.')
            break
        block_print(f'Measuring Batch {batch_num} / {total_batches}...')
        batch_list = list(islice(ct_iterator, args.batch_size))
        batch = SegStats(batch_list, folders=True)
        batch.process_segmentations()
        del batch
        batch_num += 1
        
if __name__ == '__main__':
    main()
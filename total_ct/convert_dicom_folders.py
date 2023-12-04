import argparse

from itertools import islice
from math import ceil 

from total_ct.ctbatch import CTBatch
from total_ct.utils import block_print

parser = argparse.ArgumentParser('Dicom to Nifti Tool - Only converts folders to Nifti')
parser.add_argument('--input-file', '-i', type=str, help='The list of dicom folders to convert to nifti format')
parser.add_argument('--output-dir', '-o', type=str, help='The directory to store the new nifti files (subfolders will be created')
parser.add_argument('--batch-size', '-b', type=int, default=20, help='The batch size to convert folders to nifti')
parser.add_argument('--num-workers', '-n', type=int, default=8, help='The number of processes to run to convert the folders')
parser.add_argument('--tracker-file', '-t', type=str, default=None, help='The tracker file to log completed directories. Default will be the name of the input file.log')
args = parser.parse_args()

def main():
    args.save_copy = True
    if args.tracker_file is None:
        args.tracker_file = f'{args.input_file.replace(".txt", ".log")}'
    if args.output_dir[-1] == '/':
        args.output_dir = args.output_dir[:-1]
        
    with open(args.input_file) as f:
        all_dicom_folders = f.read().splitlines()
        dicom_iterator = iter(all_dicom_folders)
        
    # Give a quick overview
    total_batches = ceil(len(all_dicom_folders) / args.batch_size)
    block_print(
        f'Total Number of CT Scans: {len(all_dicom_folders)}',
        f'Batch Size: {args.batch_size}',
        f'Total Number of Batches: {total_batches}'
    )
    
    batch_num = 0
    while True:
        if batch_num > total_batches:
            break
        block_print(f'Predicting Batch {batch_num} / {total_batches}...')
        # Get a list of inputs batch_size long
        batch_list = list(islice(dicom_iterator, args.batch_size))
        batch = CTBatch(batch_list, args, args.num_workers)
        batch.set_up_batch()
        with open(args.tracker_file, 'a') as f:
            for file in batch_list:
                f.write(f'{file}\n')
        batch_num += 1
    
if __name__ == '__main__':
    main()
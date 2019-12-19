#!/usr/bin/env python

"""
Converts the SDO dataset to numpy memory mapped objects, so that PyTorch dataloaders do
much less work when fetching data to speed up training. To run this script faster, it
is recommended that you start a machine with 12 virtual cores.

To use, simply call this script, which already has the years to convert inside of it:

./scripts/convert_to_mm.py

Once this has run remember to also run create_mm_inventory.py.py to generate a Pandas
inventory pickle file.
"""
import argparse
from multiprocessing import Pool
import pathlib
import sys
import glob
import os

import numpy as np

import sdo.io
from sdo.global_vars import DATA_BASEDIR


def convert_files2mm(details, resolution=512):
    y, m, d, files = details
    print(y, m, d, len(files))
    for f in files:
        # Note: np.memmap in write mode returns a file pointer; when we change the file
        # pointer this will change and save its contents to disk.
        new_filename = f.replace('SDOML', 'SDOMLmm').replace('.npz', '.mm')
        new_path = os.path.dirname(os.path.abspath(new_filename))
        pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
        fp = np.memmap(new_filename, dtype=np.float32, mode='w+', shape=(resolution, resolution))
        fp[:, :] = (np.load(f))['x']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert *.npz files to *.mm files')
    parser.add_argument('-y','--years', nargs='+', type=int,
                        help='<Required> Years to convert',
                        required=True)
    args = parser.parse_args()
    print('Handling years: {}'.format(args.years))

    for y in args.years:
        print('Convert year {}...'.format(y))
        for m in range(1, 13):
            tasks = []
            for d in range(1, 32):
                data_path = os.path.join(
                    DATA_BASEDIR, "{0:04d}/{1:02d}/{2:02d}/*npz".format(y, m, d))
                files = glob.glob(data_path)
                tasks.append((y, m, d, files))

            # TODO: Determine num of vcores on machines dynamically.
            with Pool(processes=12) as p:
                p.map(convert_files2mm, tasks)

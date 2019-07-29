#!/usr/bin/env python

"""
Converts the SDO dataset to numpy memory mapped objects, so that PyTorch dataloaders do
much less work when fetching data to speed up training.

To use, call this file with each year that you would like to change, calling it
multiple times with the years you would like to convert. Example:

./scripts/convert_to_mm.py 2010
./scripts/convert_to_mm.py 2011

"""
import sys
import numpy as np
import sdo.io
import glob

def convert_files2mm(files, resolution=512):
    for f in files:
        # Note: np.memmap in write mode returns a file pointer; when we change the file
        # pointer this will change and save its contents to disk.
        fp = np.memmap(f.replace('SDOML', 'SDOMLmm').replace('.npz', '.mm'), 
                      dtype=np.float32, mode='w+', shape=(resolution, resolution))
        fp[:, :] = (np.load(f))['x']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        print('Processing :', year)

    for m in range(1, 13):
        for d in range(1, 32):
            files = glob.glob('/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML/{0:04d}/{1:02d}/{2:02d}/*npz'.format(year, m, d))
            print(year, m, d, len(files))
            convert_files2mm(files)

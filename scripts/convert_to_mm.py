#!/usr/bin/env python

"""
Converts the SDO dataset to numpy memory mapped objects, so that PyTorch dataloaders do
much less work when fetching data to speed up training.

To use, simply call this script, which already has the years to convert inside of it:

./scripts/convert_to_mm.py

Once this has run remember to also run ./scripts/convert_to_mm.py to generate a Pandas
inventory pickle file.
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
    years = [2010,
             2011,
             2012,
             2013,
             2014,
             2015,
             2016,
             2017,
             2018]

    for y in years:
        print('Convert year {}...'.format(y))
        for m in range(1, 13):
            for d in range(1, 32):
                files = glob.glob('/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML/{0:04d}/{1:02d}/{2:02d}/*npz'.format(y, m, d))
                print(y, m, d, len(files))
                convert_files2mm(files)

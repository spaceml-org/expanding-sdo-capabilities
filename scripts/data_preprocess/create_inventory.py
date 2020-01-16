#!/usr/bin/env python

"""
Given SDO data that has been converted to Numpy memory mapped format via scripts/convert_to_mm.py, generates
a corresponding Pandas inventory Pickle file.
"""

import sys
import glob

import pandas as pd
import numpy as np

from sdo.global_vars import (
    DATA_BASEDIR,
    DATA_INVENTORY,
    )


def sdoml_inventory(dir=DATA_BASEDIR, inv=DATA_INVENTORY):
    """
    Purpose: Create inventory file (pandas df) of SDOML files on local disk
    """
    files = glob.glob(dir+'/*/*/*/*.mm') 

    years  = np.zeros(shape=(len(files),), dtype='uint')
    months = np.zeros(shape=(len(files),), dtype='uint8')
    days   = np.zeros(shape=(len(files),), dtype='uint8')
    channels= np.zeros(shape=(len(files),), dtype=object)
    hours  = np.zeros(shape=(len(files),), dtype='uint8')
    minutes= np.zeros(shape=(len(files),), dtype='uint8')
    index = np.zeros(shape=(len(files),), dtype='int')

    for i in range(len(files)):
        p = parse_name(files[i])
        channels[i] = p[0]
        years[i] = p[1]
        months[i] = p[2]
        days[i] = p[3]
        hours[i]= p[4]
        minutes[i] = p[5]
        index[i] = (((int(p[1])*12 + int(p[2]))*31 + int(p[3]))*24 + int(p[4]))*60 + int(p[5])
        if i % 1000000 == 0:
            print("Done",i)
    df = pd.DataFrame()
    df['year'] = years
    df['month'] = months
    df['day'] = days
    df['hour'] = hours
    df['min'] = minutes
    df['channel'] = channels
    df['index'] = index
    df['file'] = files
    df = df.sort_values('index')
    df = df.set_index('index')
    df.to_pickle(inv)


def parse_name(file):
    """
    Purpose: Parse SDOML filename
    """
    b = file.split('AIA')
    if len(b) != 2:
        b = file.split('HMI')[1]
    else: 
        b = b[1]
    a = b.split('_')
    wave = a[2].split('.')[0]
    year, month, day = a[0][0:4], a[0][4:6], a[0][6:8]
    hour, minute = a[1][0:2], a[1][2:4]
    return wave, year, month, day, hour, minute


if __name__ == '__main__':
    sdoml_inventory()
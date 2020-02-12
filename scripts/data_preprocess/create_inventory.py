#!/usr/bin/env python
"""
Given SDO data that is in *.npz format, generates a corresponding Pandas inventory Pickle file.
Before running this file you should make sure combine_channels.py was run to generate
combined 'all' channel files per timestamp.
"""
from functools import reduce
from multiprocessing.pool import ThreadPool
import os
import sys
import glob

from google.cloud import storage

import numpy as np

import pandas as pd


# Bucket name of where we store our SDO data.
GCP_BUCKET_NAME = 'fdl-sdo-data'

# Root prefix for GCP SDO data.
GCP_ROOT_PREFIX = 'SDOMLnpz'

# Path to NPZ inventory for quick look ups.
GCP_INVENTORY_PATH = 'SDOMLnpz/inventory.pkl'

# File endings we want to process.
VALID_FILE_ENDINGS = {
    '.npz': True,
    '.pklz': True,
}

# Years to process.
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]


def connect_gcp(gcp_bucket_name):
    """
    Connect to the Google Cloud Provider storage bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(gcp_bucket_name)
    return client, bucket


def list_data_files(bucket):
    # Process each year in parallel across IO, as otherwise its just too slow to
    # enumerate the huge number of files.
    pool = ThreadPool(len(YEARS))
    multi_results = pool.map(lambda year: list_data_files_for_year(bucket, year), YEARS)

    # Efficiently combine all the results together, as these are very big arrays.
    total_len = reduce(lambda acc, update_val: acc + len(update_val), multi_results, 0)    
    results = [None] * total_len
    start_idx = 0
    for arr in multi_results:
        results[start_idx:start_idx+len(arr)] = arr
        start_idx += len(arr)

    print('Finished enumerating all files for all years\n')
    return results
    
def list_data_files_for_year(bucket, year):
    """
    List all the data files we need to process into the inventory.
    """
    print('Enumerating available filenames on GCP for year {}...'.format(year))
    results = []
    for i, b in enumerate(bucket.list_blobs(prefix=get_year_path(year))):
        filename, ext = os.path.splitext(b.name)
        if ext not in VALID_FILE_ENDINGS:
            print('Skipping {} for year {}'.format(b.name, year))
            continue
        results.append(b.name)
        if i % 100000 == 0:
            print('Enumerated available filename chunks {} for year {}'.format(i, year))
    print('{} files available for year {}'.format(len(results), year))
    return results


def get_year_path(year):
        return str('{}/{}'.format(GCP_ROOT_PREFIX, year))

    
def generate_inventory(files):
    """
    Purpose: Create inventory file (pandas df) of SDOML files on Google
    Cloud Buckets.
    """    
    print('Generating inventory for {} files...'.format(len(files)))

    years  = np.zeros(shape=(len(files),), dtype='uint')
    months = np.zeros(shape=(len(files),), dtype='uint8')
    days   = np.zeros(shape=(len(files),), dtype='uint8')
    channels = np.zeros(shape=(len(files),), dtype=object)
    hours  = np.zeros(shape=(len(files),), dtype='uint8')
    minutes = np.zeros(shape=(len(files),), dtype='uint8')
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
            print("Done", i)
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
    
    print('Finished generating inventory\n')
    return df


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


def upload_inventory(df):
    print('Uploading inventory...')
    local_path = '/tmp/inventory.pkl'
    print('Saving inventory to local file...')
    df.to_pickle(local_path, compression='gzip')
    blob = storage.Blob(GCP_INVENTORY_PATH, bucket)
    print('Uploading inventory to GCP...')
    blob.upload_from_filename(local_path)
    print('Inventory uploaded to {}'.format(GCP_INVENTORY_PATH))


if __name__ == '__main__':
    client, bucket = connect_gcp(GCP_BUCKET_NAME)
    files = list_data_files(bucket)
    df = generate_inventory(files)
    upload_inventory(df)
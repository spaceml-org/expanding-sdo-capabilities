"""
In this module we define a PyTorch SDO dataset.
"""
import bz2
from math import ceil
import logging
from io import BytesIO
from os import path
import pickle
import random
import re
import time
from threading import Thread

from google.cloud import storage
from google.cloud.storage.blob import Blob

import numpy as np

import pandas as pd

import torch
from torch.utils.data import Dataset

from sdo.io import sdo_find, sdo_scale
from sdo.pytorch_utilities import to_tensor
from sdo.ds_utility import minmax_normalization

_logger = logging.getLogger(__name__)


# TODO!!! Have this also still work with NAS file paths.
# TODO!!! Benchmark how well this works with the new combined timestamp files.
class SDO_Dataset(Dataset):
    """
    Custom Dataset class compatible with torch.utils.data.DataLoader.
    It can be used to flexibly load a train or test dataset from the SDO local folder,
    asking for a specific range of years and a specific frequency in months, days, hours,
    minutes. Scaling is applied by default, normalization can be optionally applied.

    Note that the GCP storage bucket portion depends on the scripts combine_channels.py
    and create_inventory.py having been run before to pre-optimize loading all the channels
    per timestep.
    """

    def __init__(
        self,
        gcp_project,
        gcp_bucket_name,
        inventory_path,
        batch_size=1,
        instr=["AIA", "AIA", "HMI"],
        channels=["0171", "0193", "bz"],
        yr_range=[2010, 2018],
        mnt_step=1,
        day_step=1,
        h_step=6,
        min_step=60,
        resolution=512,
        subsample=1,
        test=False,
        test_ratio=0.3,
        shuffle=False,
        normalization=0,
        scaling=True,
        holdout=False,
    ):
        """
        Args:
            gcp_project (str): Google Cloud Provider project name, such as 'space-weather-sdo'.
            gcp_bucket_name (str): Google Cloud Provider bucket name where data files are located,
                such as 'fdl-sdo-data'.
            inventory_path (str|bool): path on GCP to a pre-computed inventory file that contains
                a dataframe of existing files. If False (or not valid) the file search is done
                by folder and it is much slower.
            batch_size (int): Batch size of results returned from __getitem__; we download the data
                in parallel inside __getitem__ for efficiency reasons.
            channels (list string): channels to be selected
            instr (list string): instrument to which each channel corresponds to. 
                                 It has to be of the same size of channels.
            yr_range (list int): range of years to be selected
            mnt_step (int): month frequency
            day_step (int): day frequency
            h_step (int): hour frequency
            min_step (int): minute frequency
            resolution (int): original resolution
            base_dir (str): path to the main dataset folder
            test (bool): if True, a test dataset is returned. By default the training dataset is returned.
            test_ratio (float): percentage of data to be used for testing. Training ratio is 1-test_ratio.
            shuffle (bool): if True, the dataset will be shuffled. Keep it False if you want to return
                            a time-ordered dataset.
            subsample (int): if 1 resolution of the final images will be as the original. 
                             If > 1 the image is downsampled. i.e. if resolution=512 and 
                             subsample=4, the images will be 128*128
            normalization (int): if 0 normalization is not applied, if > 0 a normalization
                                 by image is applied (only one type of normalization implemented 
                                 for now)
            scaling (bool): if True pixel values are scaled by the expected max value in active regions
                            (see sdo.io.sdo_scale)
            holdout (bool): if True use the holdout as test set. test_ratio is ignored in this case.
        """
        assert day_step > 0 and h_step > 0 and min_step > 0

        self.gcp_project = gcp_project
        self.gcp_bucket_name = gcp_bucket_name
        self.batch_size = batch_size
        self.instr = instr
        self.channels = channels
        self.resolution = resolution
        self.subsample = subsample
        self.shuffle = shuffle
        self.yr_range = yr_range
        self.mnt_step = mnt_step
        self.day_step = day_step
        self.h_step = h_step
        self.min_step = min_step
        self.test = test
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.scaling = scaling
        self.holdout = holdout
        if inventory_path:
            self.inventory_path = inventory_path
        else:
            _logger.warning("A valid inventory file has NOT be passed"
                            "If this is not expected check the path.")
            self.inventory_path = False

        # TODO self.timestamps is not used in get_item
        self.files, self.timestamps = self.create_list_files()

    def find_months(self):
        "select months for training and test based on test ratio"
        # November and December are kept as holdout
        if not self.holdout:
            months = np.arange(1, 11, self.mnt_step)
            if self.test:
                n_months = int(len(months) * self.test_ratio)
                months = months[-n_months:]
                _logger.info('Testing on months "%s"' % months)
            else:
                n_months = int(len(months) * (1 - self.test_ratio))
                months = months[:n_months]
                _logger.info('Training on months "%s"' % months)
        else:
            n_months = [11, 12]
        return months

    def create_list_files(self):
        """
        Find path to files that correspond to the requested timestamps. A timestamp
        is returned only if the files from ALL the requested channels are found.

        Returns: list of lists of strings, list of tuples. The first argument are the 
             path to the files, each row is a timestamp. The second argument are the
             correspondant timestamps.

        """
        _logger.info('Loading SDOML from GCP bucket "%s"' % self.gcp_bucket_name)
        _logger.info('Loading SDOML inventory file from "%s"' % self.inventory_path)
        client, bucket = connect_gcp(self.gcp_project, self.gcp_bucket_name)
        indexes = ['year', 'month', 'day', 'hour', 'min']
        yrs = np.arange(self.yr_range[0], self.yr_range[1]+1)
        months = self.find_months()
        days = np.arange(1, 32, self.day_step)
        hours = np.arange(0, 24, self.h_step)
        minus = np.arange(0, 60, self.min_step)
        tot_timestamps = np.prod([len(x) for x in [yrs, months, days, hours, minus]])
        _logger.debug("Timestamps requested values: ")
        _logger.debug("Years: %s" % ','.join('{}'.format(i) for i in (yrs)))
        _logger.debug("Months: %s" % ','.join('{}'.format(i) for i in (months)))
        _logger.debug("Days: %s" % ','.join('{}'.format(i) for i in (days)))
        _logger.debug("Hours: %s" % ','.join('{}'.format(i) for i in (hours)))
        _logger.debug("Minutes: %s" % ','.join('{}'.format(i) for i in (minus)))
        _logger.info("Max number of timestamps: %d" % tot_timestamps)

        if self.inventory_path:
            inventory_data = bucket.blob(self.inventory_path).download_as_string()
            df = pd.read_pickle(BytesIO(inventory_data), compression='gzip')
            cond0 = df['channel'].isin(self.channels)
            cond1 = df['year'].isin(yrs)
            cond2 = df['month'].isin(months)
            cond3 = df['day'].isin(days)
            cond4 = df['hour'].isin(hours)
            cond5 = df['min'].isin(minus)

            sel_df = df[cond0 & cond1 & cond2 & cond3 & cond4 & cond5]
            n_sel_timestamps = sel_df.groupby(indexes).head(1).shape[0]
            _logger.info("Timestamps found in the inventory: %d (%.2f)" % 
                         (n_sel_timestamps, float(n_sel_timestamps)/tot_timestamps))
            grouped_df = sel_df.groupby(indexes).size()
            # we select only timestamp that have files for all the channels
            grouped_df = grouped_df[grouped_df == len(self.channels)].to_frame()
            sel_df = sel_df.reset_index().drop('index', axis=1)
            sel_df = pd.merge(grouped_df, sel_df, how='inner',
                              left_on=indexes, right_on=indexes)
            # sorting is essential, the order of the channels must be consistent
            s_files = sel_df.sort_values('channel').groupby(indexes)['file'].apply(list)
            files = s_files.values.tolist()
            timestamps = s_files.index.tolist()
            discarded_tm = n_sel_timestamps - len(timestamps)
        else:
            _logger.warning(
                'A valid inventory file has not been passed in, be prepared to wait.')
            files = []
            timestamps = []
            discarded_tm = 0
            for y in yrs:
                for month in months:
                    for d in days:
                        for h in hours:
                            for minu in minus:
                                # if a single channel is missing for the combination
                                # of parameters result is -1
                                # TODO: Get this working against listing the files in GCP buckets
                                # directories.
                                result = sdo_find(y, month, d, h, minu,
                                                  initial_size=self.resolution,
                                                  basedir=self.data_basedir,
                                                  instrs=self.instr,
                                                  channels=self.channels,
                                                  )
                            if result != -1:
                                files.append(result)
                                timestamp = (y, month, d, h, minu)
                                timestamps.append(timestamp)
                            else:
                                discarded_tm += 1
        if len(files) == 0:
            _logger.error("No input images found")
        else:
            _logger.info("N timestamps discarded because channel is missing = %d (%.5f)" % 
                         (discarded_tm, float(discarded_tm)/n_sel_timestamps))
            _logger.info("Selected timestamps = %d" % len(files))
            _logger.info("N images = %d" % (len(files)*len(self.channels)))
            if self.shuffle:
                _logger.warning(
                    "Shuffling is being applied, this will alter the time sequence.")
                random.shuffle(files)
        return files, timestamps

    def __len__(self):
        return ceil(len(self.files) / float(self.batch_size))

    def __getitem__(self, batch_index):
        """
        Download all of the items in our batch in parallel.
        """
        start_idx = batch_index * self.batch_size
        last_batch = batch_index == (len(self) - 1)
        if last_batch:
            end_idx = len(self.files) % self.batch_size
        else:
            end_idx = start_idx + (self.batch_size - 1)

        #print('batch_size: {}, start_idx: {}, end_idx: {}, last_batch: {}'.format(self.batch_size, start_idx,
        #                                                                          end_idx, last_batch))
        item_paths = self.files[start_idx:end_idx+1]

        # TODO!!! Have these threads stay around and each keep their own GCP connection long-lived.
        threads = [None] * self.batch_size
        results = [None] * self.batch_size
        for i in range(self.batch_size):
            threads[i] = Thread(target=download_item,
                                args=(results, i, item_paths[i],
                                      self.gcp_project, self.gcp_bucket_name,
                                      self.resolution, self.subsample,
                                      self.channels, self.scaling,
                                      self.normalization))
            threads[i].start()

        # Wait for all the threads to finish downloading and transforming their
        # results.
        for i in range(self.batch_size):
            threads[i].join()

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return to_tensor(results, dtype=torch.float)


# This runs in a thread, so we place it outside of the class to make it clear
# what data it can interact with.
def download_item(item_results, item_idx, item_paths, gcp_project, gcp_bucket_name, resolution,
                  subsample, channels, scaling, normalization):
    """
    This function will return a single row of the dataset, where each image has 
    been scaled and normalized if requested in the class initialization.
    Args:
        item_results (List[numpy]): List of numpy images with the results to make it
                                    easy to return results to the parent thread.
        item_idx (int): Where to place the results for this thread inside item_results.
        item_paths (str[]): GCP paths for each of the channels for this item.
        gcp_project (str): GCP project containing the GCP bucket.
        gcp_bucket_name (str): Path to the GCP bucket to work with.
        resolution (int): original resolution.
        subsample (int): if 1 resolution of the final images will be as the original. 
                         If > 1 the image is downsampled. i.e. if resolution=512 and 
                         subsample=4, the images will be 128*128
        channels (list string): channels to be selected
        scaling (bool): if True pixel values are scaled by the expected max value in active regions
                        (see sdo.io.sdo_scale)
        normalization (int): if 0 normalization is not applied, if > 0 a normalization
                             by image is applied (only one type of normalization implemented 
                             for now)

    Returns: numpy array.
    """
    wait_time_s = 1
    # Continue forever until we get our data.
    # UNCOMMENT!!!
    #while True:
    filename = get_combined_path(item_paths[0])
    try:
        client, bucket = connect_gcp(gcp_project, gcp_bucket_name)        
        blob = bucket.get_blob(filename) # REVISIT!!!
        if not blob:
            # TODO!!! Make this cleaner.
            print('No blob for {}'.format(filename))
            return
        # TODO!!! This is failing, figure out why.
        all_channels = pickle.loads(bz2.decompress(blob.download_as_string()))
        print("all_channels: {}".format(all_channels))

        n_channels = len(channels)
        size = int(resolution / subsample)
        # the original images are NOT bytescaled
        # we directly convert to 32 because the pytorch tensor will need to be 32
        item = np.zeros(shape=(n_channels, size, size), dtype=np.float32)

        for c, channel_name in enumerate(channels):
            assert channel_name in all_channels, '{} not in the combined timestamps!'.format(channel_name)
            img_data = all_channels[channel_name]
            print('channel_name: {}, value: {}'.format(channel_name, img_data)) # REMOVE!!!
            img = np.load(BytesIO(img_data), allow_pickle=True)['x']
            if subsample > 1:
                # Use numpy trick to essentially downsample the full resolution image by 'subsample'.
                img = img[::subsample, ::subsample]
            if scaling:
                # divide by roughly the mean of the channel
                img = sdo_scale(img, channel)
            if normalization > 0:
                if normalization == 1:
                    img = minmax_normalization(img)
                else:
                    _logger.error("This type of normalization is not implemented."
                                  "Original image is returned")
            item[c] = img

        item_results[item_idx] = item
        return
    except Exception as e:
        _logger.error("Exception for {}: {}".format(filename, e))
        # UNCOMMENT!!!
#         time.sleep(wait_time_s * random.random())
#         wait_time_s *= 2


def get_combined_path(channel_path):
    """
    Given a path to a channel, generates an *_all.pklz version of that path pointing
    to the combined channels file.
    """
    return re.sub(r'_[^._]*?\.npz', '_all.pklz', channel_path)


def connect_gcp(gcp_project, gcp_bucket_name):
    """
    Connect to the Google Cloud Provider storage bucket.
    """
    client = storage.Client(gcp_project)
    bucket = client.get_bucket(gcp_bucket_name)
    _logger.info('Connected to GCP storage bucket {}'.format(gcp_bucket_name))
    return client, bucket
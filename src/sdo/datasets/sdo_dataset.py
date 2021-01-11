"""
In this module we define a pytorch SDO dataset
"""
import logging
from os import path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sdo.io import sdo_find, sdo_scale
from sdo.pytorch_utilities import to_tensor
from sdo.ds_utility import minmax_normalization

_logger = logging.getLogger(__name__)


class SDO_Dataset(Dataset):
    """ Custom Dataset class compatible with torch.utils.data.DataLoader.
    It can be used to flexibly load a train or test dataset from the SDO local folder,
    asking for a specific range of years and a specific frequency in months, days, hours,
    minutes. Scaling is applied by default, normalization can be optionally applied. """

    def __init__(
        self,
        data_basedir,
        data_inventory,
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
        apodize=False,
        holdout=False,
        mm_files=True,
    ):
        """
        Args:
            data_basedir (str): path to locate training/testing data.
            data_inventory (str): path to a pre-computed inventory file that contains
                a dataframe of existing files. If False(or not valid) the file search is done
                by folder and it is much slower.
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
            apodize (bool): if True it masks the Sun’s limb. Remove anything farther than 1 solar radii from the center.
            mm_files (bool): if True it loads memory maps format data. If False it loads npz format data. SDOML available
            online is usually in npz format.
        """
        assert day_step > 0 and h_step > 0 and min_step > 0

        self.data_basedir = data_basedir
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
        self.apodize = apodize
        self.holdout = holdout
        self.mm_files = mm_files

        _logger.info("apodize={}".format(self.apodize))

        if path.isfile(data_inventory):
            self.data_inventory = data_inventory
        else:
            _logger.warning("A valid inventory file has NOT be passed"
                            "If this is not expected check the path.")
            self.data_inventory = False
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
            months = [11, 12]
        return months

    def create_list_files(self):
        """
        Find path to files that correspond to the requested timestamps. A timestamp
        is returned only if the files from ALL the requested channels are found.

        Returns: list of lists of strings, list of tuples. The first argument are the 
             path to the files, each row is a timestamp. The second argument are the
             correspondant timestamps.

        """
        _logger.info('Loading SDOML from "%s"' % self.data_basedir)
        _logger.info('Loading SDOML inventory file from "%s"' % self.data_inventory)
        indexes = ['year', 'month', 'day', 'hour', 'min']
        yrs = np.arange(self.yr_range[0], self.yr_range[1] + 1)
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

        if self.data_inventory:
            df = pd.read_pickle(self.data_inventory)
            cond0 = df['channel'].isin(self.channels)
            cond1 = df['year'].isin(yrs)
            cond2 = df['month'].isin(months)
            cond3 = df['day'].isin(days)
            cond4 = df['hour'].isin(hours)
            cond5 = df['min'].isin(minus)

            sel_df = df[cond0 & cond1 & cond2 & cond3 & cond4 & cond5]
            n_sel_timestamps = sel_df.groupby(indexes).head(1).shape[0]
            _logger.info("Timestamps found in the inventory: %d (%.2f)" %
                         (n_sel_timestamps, float(n_sel_timestamps) / tot_timestamps))
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
            n_sel_timestamps = 0
            discarded_tm = 0
            for y in yrs:
                for month in months:
                    for d in days:
                        for h in hours:
                            for minu in minus:
                                # if a single channel is missing for the combination
                                # of parameters result is -1
                                result = sdo_find(y, month, d, h, minu,
                                                  initial_size=self.resolution,
                                                  basedir=self.data_basedir,
                                                  instrs=self.instr,
                                                  channels=self.channels,
                                                  )
                                n_sel_timestamps += n_sel_timestamps
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
                         (discarded_tm, float(discarded_tm) / n_sel_timestamps))
            _logger.info("Selected timestamps = %d" % len(files))
            _logger.info("N images = %d" % (len(files) * len(self.channels)))
            if self.shuffle:
                _logger.warning(
                    "Shuffling is being applied, this will alter the time sequence.")
                indices = np.arange(len(files))
                random.shuffle(indices)
                tmp_files = []
                tmp_timestamps = []
                for i in indices:
                    tmp_files.append(files[i])
                    tmp_timestamps.append(timestamps[i])
                files = tmp_files
                timestamps = tmp_timestamps
        return files, timestamps

    def normalize_by_img(self, img, norm_type):
        if norm_type == 1:
            return minmax_normalization(img)
        else:
            _logger.error("This type of normalization is not implemented."
                          "Original image is returned")
            return img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        This function will return a single row of the dataset, where each image has 
        been scaled and normalized if requested in the class initialization.
        Args:
            index (int): dataset row index

        Returns: pytorch tensor

        """
        size = int(self.resolution / self.subsample)
        n_channels = len(self.channels)
        # the original images are NOT bytescaled
        # we directly convert to 32 because the pytorch tensor will need to be 32
        item = np.zeros(shape=(n_channels, size, size), dtype=np.float32)

        img = np.zeros(shape=(size, size), dtype=np.float32)
        for c in range(n_channels):
            if self.mm_files: # Load the SDOML files depending on which extension used. mm_file = true will load memory maps.
                temp = np.memmap(self.files[index][c], shape=(self.resolution, self.resolution), mode='r', dtype=np.float32)
            else:
                temp = np.load(self.files[index][c], allow_pickle=True)['x']
            img[:, :] = temp[::self.subsample, ::self.subsample]
            if self.scaling:
                # divide by roughly the mean of the channel
                img = sdo_scale(img, self.channels[c])
            if self.normalization > 0:
                img = self.normalize_by_img(img, self.normalization)
            item[c, :, :] = img

        if self.apodize:
            # Set off limb pixel values to zero
            x = np.arange((img.shape[0]), dtype=np.float) - img.shape[0] / 2 + 0.5
            y = np.arange((img.shape[1]), dtype=np.float) - img.shape[1] / 2 + 0.5
            xgrid = np.ones(shape=(img.shape[1], 1)) @ x.reshape((1, x.shape[0]))
            ygrid = y.reshape((y.shape[0], 1)) @ np.ones(shape=(1, img.shape[0]))
            dist = np.sqrt(xgrid * xgrid + ygrid * ygrid)
            mask = np.ones(shape=dist.shape, dtype=np.float)
            mask = np.where(dist < 200. / self.subsample, mask,
                            0.0)  # Radius of sun at 1 AU is 200*4.8 arcsec
            for c in range(len(self.channels)):
                item[c, :, :] = item[c, :, :] * mask

        timestamps = self.timestamps[index]
        output = [to_tensor(item), to_tensor(timestamps)]
   
        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return output
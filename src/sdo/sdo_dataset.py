"""
In this module we define a pytorch SDO dataset
"""
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from sdo.global_vars import BASEDIR
from sdo.io import sdo_find, sdo_bytescale
from sdo.pytorch_utilities import to_tensor
from sdo.ds_utility import minmax_normalization
import logging

import pdb


_logger = logging.getLogger(__name__)


class SDO_Dataset(Dataset):
    """ Custom Dataset class compatible with torch.utils.data.DataLoader. 
    It can be used to flexibly load a train or test dataset from the SDO local folder,
    asking for a specific range of years and a specific frequency in months, days, hours,
    minutes. Normalization and bytescaling can be applied. """

    def __init__(
        self,
        device,
        instr=["AIA", "AIA", "HMI"],
        channels=["0171", "0193", "bz"],
        yr_range=[2010, 2018],
        mnt_step=1,
        day_step=1,
        min_step=6,
        resolution=512,
        subsample=1,
        base_dir=BASEDIR,
        test=False,
        test_ratio=0.3,
        shuffle=True,
        normalization=1,
        bytescaling=True,
        shuffle_seed=1234,
    ):
        """

        Args:
            device (torch.device): device where to send the data
            channels (list string): channels to be selected
            instr (list string): instrument to which each channel corresponds to. 
                                 It has to be of the same size of channels.
            yr_range (list int): range of years to be selected
            mnt_step (int): month frequence
            day_step (int): day frequence
            min_step (int): minute frequence
            resolution (int): original resolution
            base_dir (str): path to the main dataset folder
            test (bool): if True, a test dataset is returned. By default the training dataset is returned.
            test_ratio (float): percentage of data to be used for testing. Training ratio is 1-test_ratio.
            subsample (int): if 1 resolution of the final images will be as the original. 
                             If > 1 the image is downsampled. i.e. if resolution=512 and 
                             subsample=4, the images will be 128*128
            shuffle (bool): if True, the dataset will be shuffled
            normalization (int): if 0 normalization is not applied, if >  0 a normalization
                                 by image is applied (only one type of normalization implemented 
                                 for now)
            bytescaling (bool): if True pixel values are scaled for 8-bit display
            shuffle_seed (int): seed to be used when shuffling the rows of the dataset. 
                                if shuffle=False this is ignored

        """
        self.device = device
        self.dir = base_dir
        self.instr = instr
        self.channels = channels
        self.resolution = resolution
        self.subsample = subsample
        self.shuffle = shuffle
        self.yr_range = yr_range
        self.mnt_step = mnt_step
        self.day_step = day_step
        self.min_step = min_step
        self.test = test
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.bytescaling = bytescaling
        self.shuffle_seed = shuffle_seed
        self.files = self.create_list_files()

    def find_months(self):
        months = np.arange(1, 13, self.mnt_step)
        if self.test:
            n_months = int(len(months) * self.test_ratio)
            months = months[-n_months:]
        else:
            n_months = int(len(months) * (1 - self.test_ratio))
            months = months[:n_months]
        _logger.info('Running on months "%s"' % months)
        return months

    def create_list_files(self):
        """

        Returns: list of strings

        """
        _logger.info('Loading SDOML from "%s"' % self.dir)
        files = []
        months = self.find_months()
        for y in np.arange(self.yr_range[0], self.yr_range[1]):
            for m in months:
                for d in np.arange(1, 32, self.day_step):
                    for h in np.arange(0, 24, self.min_step):
                        # if a single channel is missing for the combination
                        # of parameters result is -1
                        result = sdo_find(y, m, d, h, 0,
                                          instrs=self.instr,
                                          channels=self.channels,
                                          basedir=self.dir,
                                          )
                        if result != -1:
                            files.append(result)
        if len(files) == 0:
            _logger.error("No input images found")
        else:
            _logger.info("Number of SDO files = %d" % len(files))
            if self.shuffle:
                random.seed(self.shuffle_seed)
                random.shuffle(files)
        return files

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
        been normalized and bytescaled if requested in the class initialization.
        Args:
            index (int): dataset row index

        Returns: pytorch tensor

        """
        size = int(self.resolution / self.subsample)
        n_channels = len(self.channels)
        # the original images are NOT bytescaled
        # we directly convert to 32 because the pytorch tensor will need to be 32
        item = np.zeros(shape=(size, size, n_channels), dtype=np.float32)
        for i in range(n_channels):
            img = np.load(self.files[index][i])["x"][
                ::self.subsample, ::self.subsample]
            if self.normalization > 0:
                img = self.normalize_by_img(img, self.normalization)
            if self.bytescaling:
                item[:, :, i] = sdo_bytescale(img, self.channels[i])
            else:
                item[:, :, i] = img
        if n_channels == 1:
            item = item[np.newaxis, :, :]  # HW => CHW
        else:
            item = item.transpose([2, 0, 1])  # HWC => CHW
        tensor = to_tensor(item)
        return tensor.to(device=self.device, dtype=torch.float)

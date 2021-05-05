"""
A version of the SDO dataset that loads all the months of degraded data to reconstruct the degradation curve through time
"""
import logging
import torch
from sdo.datasets.mock_data import create_noise_image
from sdo.datasets.sdo_dataset import SDO_Dataset
import numpy as np
from sdo.io import sdo_find, sdo_scale
from sdo.pytorch_utilities import to_tensor

_logger = logging.getLogger(__name__)

class DegradationSDO_Dataset(SDO_Dataset):
    def __init__(self, *args, **kwargs):
        super(DegradationSDO_Dataset, self).__init__(*args, **kwargs)
        self.mm_files = False
        self.files, self.timestamps = super(DegradationSDO_Dataset, self).create_list_files()
        
    def find_months(self):
        # Get all months of undegraded data
        months = np.arange(1, 13, self.mnt_step)
        return months

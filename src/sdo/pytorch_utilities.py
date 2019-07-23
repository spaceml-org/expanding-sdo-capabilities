import logging
import os
import random

import torch
import numpy as np

_logger = logging.getLogger(__name__)

_dtype = torch.float #this corresponds to float32
#_device = torch.device('cuda') # change to cuda tor un on GPU

def to_tensor(value, dtype=_dtype):
    if not torch.is_tensor(value):
        if type(value) == np.int64:
            value = torch.tensor(float(value))
        elif type(value) == np.float32:
            value = torch.tensor(float(value))
        else:
            value = torch.tensor(value)
    return value

def to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except Exception as e:
            print(e)
            raise TypeError('Cannot convert to Numpy array.')
            
def init_gpu(cuda_device=0):
    """ Use the GPU. """
    torch.backends.cudnn.enabled = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Unable to continue")

    # Force ourselves to use only one GPU.
    device = torch.device("cuda:{}".format(cuda_device))
    _logger.info("Using device {} for training, current device: {}, total devices: {}".format(
        device, torch.cuda.current_device(), torch.cuda.device_count()))
    return device

def set_seed(random_seed=1, deterministic_cuda=True):
    """
    Force runs to be deterministic and reproducible. Note that forcing CUDA to be
    deterministic can have a performance impact.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
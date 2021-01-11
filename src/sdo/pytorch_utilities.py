import logging
import multiprocessing
import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np

_logger = logging.getLogger(__name__)
_dtype = torch.float # this corresponds to float32


def to_tensor(value):
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


def init_gpu(cuda_device=None):
    """ Use the GPU. """
    torch.backends.cudnn.enabled = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Unable to continue")

    if cuda_device is None:
        # Randomly keep trying available GPU devices.
        devices = np.random.permutation(list(range(torch.cuda.device_count())))
        success = False
        for cuda_device in devices:
            _logger.info('Trying to use CUDA device {}...'.format(cuda_device))
            try:
                device_str = "cuda:{}".format(cuda_device)
                torch.cuda.set_device(device_str)
                device = torch.device(device_str)
                success = True
                break
            except Exception as error:
                _logger.exception(error)
            if not success:
                raise Exception("No CUDA device is available!")
    else:
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


def pass_seed_to_worker(worker_id):
    """
    Given some pytorch DataLoader that is spawning worker forks, this method
    will ensure that they are all given the correct random seed on
    initialization to prevent the following problem:
    https://github.com/pytorch/pytorch/issues/5059
    Keep in mind that pytorch creates and destroys these workers on _every_
    epoch, so we have to be extra careful about setting our random seeds
    so they won't repeat every epoch!
    """
    # Numpy can't have random seeds greater than 2^32 - 1.
    seed = (torch.initial_seed() // (2**32 - 1)) + worker_id
    set_seed(seed)


def create_dataloader(dataset, batch_size, num_dataloader_workers, train, shuffle=True):
    assert num_dataloader_workers <= (multiprocessing.cpu_count() - 1), \
        'There are not enough CPU cores ({}) for requested dataloader ' \
        'workers ({})'.format(num_dataloader_workers, (multiprocessing.cpu_count() - 1))

    _logger.info('Using {} workers for the {} pytorch DataLoader'.format(
        num_dataloader_workers, 'training' if train else 'testing'))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_dataloader_workers,
                        # Ensure workers spawn with the right newly
                        # incremented random seed.
                        worker_init_fn=pass_seed_to_worker,
                        # Make sure that results returned from our
                        # SDO_DataSet are placed onto the GPU.
                        pin_memory=True)
    return loader

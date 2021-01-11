'''
This module contains functions for input output of the data
'''
from os import path
import numpy as np
from numpy import zeros, load
from numpy import sqrt
import logging
from sdo.global_vars import (
    DATA_FILENAME_TEMPLATE,
    B_CHANNELS,
    UV_CHANNELS,
    )

BUNIT = 2000.0  # units of 2 kGauss
AUNIT = 100.0  # units of 100 DN/s/pixel


def bytescale(c, cmin=0, cmax=255): 
    return np.clip( (c - cmin)/(cmin-cmax)*255, 0, 255).astype(np.uint8)


# the following units have been chosen based on Fig3 of Galvez et all,
# they represent the approximate mean of the channels across the full time period
AUNIT_BYCH = {'1600': 500.0, '1700': 7000.0, '0094': 10.0, '0131': 80.0, '0171': 2000.0,
               '0193': 3000.0, '0211': 1000.0, '0304': 500.0, '0335': 80.0}

_logger = logging.getLogger(__name__)


def sdo_read(year, month, day, hour, minu, basedir, instr='AIA', channel='0094',
             subsample=1):
    """
    Purpose: Find an SDOML file, and return image if it exists.
    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    minu - between 0 and 59 (note AIA data is at 6 min cadence, 
            HMI at 12 min cadence)
    basedir - directory where the SDO data set is stored.
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', 
       '0171', '0193', '0211', '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' 
       (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions

    Returns: np.array. Returns -1 if file is not found.
    """
    file = DATA_FILENAME_TEMPLATE.format(basedir, year, month, day, instr,
                                         year, month, day, hour, min,
                                         channel)
    if path.isfile(file):
        return ((load(file))['x'])[::subsample, ::subsample]
    print('{0:s} is missing'.format(file))
    return -1


def sdo_find(year, month, day, hour, minu, initial_size, basedir,
             instrs=['AIA', 'AIA', 'HMI'], channels=['0171', '0193', 'bx'], subsample=1,
             return_images=False):
    """
    Purpose: Find filenames of multiple channels of the SDOML dataset with the same 
    timestamp. 

    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    minu - between 0 and 59 (note AIA data is at 6 min cadence, HMI at 12 min cadence)
    basedir - directory where the SDO data set is stored.
    initial_size - Unscaled resolution of images.
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', '0171', '0193', '0211',
           '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' 
           (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions
    return_images (bool). If False it returns the list of files. If True
         images are returned.

    Returns: list of files if return_images False (default). 
            np.array of shape (n, n, number of channels). 
            Returns -1 if not all channels are found.
    """
    files_exist = True
    files = []
    for ind, ch in enumerate(channels):
        files.append(DATA_FILENAME_TEMPLATE.format(basedir, year, month, day,
                                                   instrs[ind], year, month,
                                                   day, hour, minu, ch))
        files_exist = files_exist*path.isfile(files[-1])
    if files_exist:
        if (not return_images):
            return files
        else:
            img = zeros(shape=(int(initial_size/subsample),
                               int(initial_size/subsample),
                               len(channels)), dtype='float32')
            for c in range(len(channels)):
                img[:, :, c] = sdo_scale((
                    (load(files[c]))['x'])[::subsample, ::subsample], channels[c])
            return img
    else:
        return -1


def sdo_bytescale(img, ch, aunit=AUNIT, bunit=BUNIT):
    """
    Purpose: Given an SDO image of a given channel, returns scaled image
    appropriate for 8-bit display (uint8)

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to
    aunit: units UV channels
    bunit: units magnetogram 

    Returns np.array of the same size of img
    """
    if ch in B_CHANNELS:
        return bytescale(img, cmin=-bunit, cmax=bunit)
    elif ch in UV_CHANNELS:
        return bytescale(sqrt(img), cmin=0, cmax=aunit)
    else:
        _logger.warning(
            "Channel not found, simply bytescaled image is returned")
        return bytescale(img)


def sdo_scale(img, ch, aunit_dict=AUNIT_BYCH, bunit=BUNIT):
    """
    Purpose: Given an SDO image of a given channel, returned scaled image.
    Scaling values are supposed to be the mean of the channel across
    the full time period.

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to
    aunit_dict (dict int): units UV channels
    bunit (int): units magnetogram 

    Returns np.array of the same size of img
    """
    if ch in B_CHANNELS:
        return img/bunit
    elif ch in UV_CHANNELS:
        return img/aunit_dict[ch]       
    else:
        _logger.error("Channel not found, input image is returned")
        return img


def sdo_reverse_scale(img, ch, aunit_dict=AUNIT, bunit=BUNIT):
    """
    Purpose: Given a scaled SDO image of a given channel, returned the unscaled image.
    Scaling values are supposed to be the mean of the channel across
    the full time period.

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to
    aunit_dict (dict int): units UV channels
    bunit (int): units magnetogram

    Returns np.array of the same size of img
    """
    aunit_dict.update((x, 1./y) for x, y in aunit_dict.items())
    bunit = 1./bunit
    return sdo_scale(img, ch, aunit=aunit_dict, bunit=bunit)


def format_epoch(epoch):
    """
    Given some epoch, expands and formats it as a string with leading zeros
    so that its suitable for prefixing to a filename, such as turning 3
    into 00003.
    """
    return str(epoch).zfill(4)


def format_graph_prefix(epoch, exp_name):
    """
    Given some epoch and experiment name, provides a prefix suitable for
    appending to saved graphs, such as 0013_some_experiment.
    """
    return "{}_{}".format(format_epoch(epoch), exp_name)
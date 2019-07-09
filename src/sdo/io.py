'''
This module contains functions for input output of the data
'''
from os import path
from numpy import zeros, load
from numpy import sqrt
from scipy.misc import bytescale
import logging
from sdo.global_vars import (BASEDIR, FILENAME_TEMPLATE, INITIAL_SIZE,
                             B_CHANNELS, UV_CHANNELS)

AUNIT = 100.0  # units of 100 DN/s/pixel
BUNIT = 2000.0  # units of 2 kGauss

_logger = logging.getLogger(__name__)


def sdo_read(year, month, day, hour, minu, instr='AIA', channel='0094',
             subsample=1, basedir=BASEDIR):
    """
    Purpose: Find an SDOML file, and return image if it exists.
    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    minu - between 0 and 59 (note AIA data is at 6 min cadence, 
            HMI at 12 min cadence)
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', 
       '0171', '0193', '0211', '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' 
       (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions
    basedir - directory where the SDO data set is stored.

    Returns: np.array. Returns -1 if file is not found.
    """
    file = FILENAME_TEMPLATE.format(basedir, year, month, day, instr, year,
                                    month, day, hour, min, channel)
    if path.isfile(file):
        return ((load(file))['x'])[::subsample, ::subsample]
    print('{0:s} is missing'.format(file))
    return -1


def sdo_find(year, month, day, hour, minu, instrs=['AIA', 'AIA', 'HMI'],
             channels=['0171', '0193', 'bx'], subsample=1, basedir=BASEDIR,
             return_images=False):
    """
    Purpose: Find filenames of multiple channels of the SDOML dataset with the same 
    timestamp. 

    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    minu - between 0 and 59 (note AIA data is at 6 min cadence, HMI at 12 min cadence)
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', '0171', '0193', '0211',
           '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' 
           (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions
    basedir - directory where the SDO data set is stored.
    return_images (bool). If False it returns the list of files. If True
         images are returned.

    Returns: list of files if return_images False (default). 
            np.array of shape (n, n, number of channels). 
            Returns -1 if not all channels are found.
    """
    files_exist = True
    files = []
    for ind, ch in enumerate(channels):
        files.append(FILENAME_TEMPLATE.format(basedir, year, month, day,
                                              instrs[ind], year, month,
                                              day, hour, minu, ch))
        files_exist = files_exist*path.isfile(files[-1])
    if files_exist:
        if (not return_images):
            return files
        else:
            img = zeros(shape=(int(INITIAL_SIZE/subsample),
                               int(INITIAL_SIZE/subsample),
                               len(channels)), dtype='float32')
            for c in range(len(channels)):
                img[:, :, c] = sdo_scale((
                    (load(files[c]))['x'])[::subsample, ::subsample], channels[c])
            return img
    else:
        return -1


def sdo_bytescale(img, ch):
    """
    Purpose: Given an SDO image of a given channel, returns scaled image
    appropriate for 8-bit display (uint8)

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to

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


def sdo_scale(img, ch, aunit=AUNIT, bunit=BUNIT):
    """
    Purpose: Given an SDO image of a given channel, returned scaled image.
    Currently a placeholder. scaling functions are subject to change based on EDA.

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to
    aunit: units UV channels
    bunit: units magnetogram 

    Returns np.array of the same size of img
    """
    if ch in B_CHANNELS:
        return img/aunit
    elif ch in UV_CHANNELS:
        return img/bunit
    else:
        _logger.error("Channel not found, input image is returned")
        return img


def sdo_reverse_scale(img, ch, aunit=AUNIT, bunit=BUNIT):
    """
    Purpose: Given a scaled SDO image of a given channel, returned the unscaled image.
    Currently a placeholder. scaling functions are subject to change based on EDA.

    Params:
    img (np.array): image to be rescaled
    ch (str): string describing the channel img belongs to
    aunit: units UV channels
    bunit: units magnetogram 

    Returns np.array of the same size of img
    """
    aunit = 1./aunit
    bunit = 1./bunit
    return sdo_scale(img, ch, aunit=aunit, bunit=bunit)

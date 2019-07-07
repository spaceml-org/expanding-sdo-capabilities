def sdo_bytescale(img, ch):
    from numpy import sqrt
    from scipy.misc import bytescale
    """
    Purpose: Given an SDO image of a given channel, returns scaled image
    appropriate for 8-bit display (uint8)
    """
    aunit = 100.0 #
    bunit = 2000.0 #units of 2 kGauss  

    if ch == 'bx':
        return bytescale(img,cmin=-bunit,cmax=bunit)
    if ch == 'by':
        return bytescale(img,cmin=-bunit,cmax=bunit)
    if ch == 'bz':
        return bytescale(img,cmin=-bunit,cmax=bunit)
    if ch == '1600':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '1700':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0094':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0131':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0171':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0193':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0211':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0304':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)
    if ch == '0335':
        return bytescale(sqrt(img),cmin=0,cmax=aunit)

    return bytescale(img)

def sdo_scale(img, ch, inverse=False):
    """
    Purpose: Given an SDO image of a given channel, returned scaled image
    currently a placeholder. scaling functions are subject to change based on EDA.
    """
    aunit = 100.0 #units of 100 DN/s/pixel
    bunit = 2000.0 #units of 2 kGauss
    
    if inverse:
        bunit = 1.0/bunit
        aunit = 1.0/aunit

    if ch == 'bx':
        return img/bunit  
    if ch == 'by':
        return img/bunit                                                                                                                                              
    if ch == 'bz':
        return img/bunit          
    if ch == '1600':
        return img/aunit
    if ch == '1700':
        return img/aunit
    if ch == '0094':
        return img/aunit                                                                                                                                       
    if ch == '0131':
        return img/aunit 
    if ch == '0171':
        return img/aunit  
    if ch == '0193':
        return img/aunit 
    if ch == '0211':
        return img/aunit 
    if ch == '0304':
        return img/aunit 
    if ch == '0335':
        return img/aunit 
    #If channel is not found, then just return original image
    return img


def sdo_read(year, month, day, hour, min, instr='AIA', channel='0094',subsample=1,basedir='/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML'):
    """
    Purpose: Find an SDOML file, and return image if it exists.
    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    min - between 0 and 59 (note AIA data is at 6 min cadence, HMI at 12 min cadence)
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions
    basedir - directory where the SDO data set is stored.
    """
    from os import path
    from numpy import load
    file = '{0:s}/{1:04d}/{2:02d}/{3:02d}/{4:s}{5:04d}{6:02d}{7:02d}_{8:02d}{9:02d}_{10:s}.npz'.format(basedir,year,month,day,instr,year,month,day,hour,min,channel)
    if path.isfile(file):
        return ((load(file))['x'])[::subsample,::subsample]
    print('{0:s} is missing'.format(file))
    return -1

def sdo_find(year, month, day, hour, min, instrs=['AIA','AIA','HMI'], channels= ['0171','0193','bx'],subsample=1,basedir='/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML', return_images=False):
    from numpy import zeros, load
    """
    Purpose: Find filenames of multiple channels of the SDOML dataset with the same timestamp. Returns -1 if not all channels are found.

    Parameters:
    year / month / day - a date between May 17 2010 to 12/31/2018
    hour - between 0 and 23
    min - between 0 and 59 (note AIA data is at 6 min cadence, HMI at 12 min cadence)
    instr - 'AIA' or 'HMI'
    channel - 
       if instr=='AIA', channel should be one of '0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700'
       if instr=='HMI', channel should be one of 'bx', 'by', 'bz' (last is the line-of-sight component of the magnetic field)
    subsample - return image with every subsample-th pixel in both dimensions
    basedir - directory where the SDO data set is stored.
    """
    from os import path
    files_exist = True
    files = []
    ind = 0
    for ch in channels:
        files.append('{0:s}/{1:04d}/{2:02d}/{3:02d}/{4:s}{5:04d}{6:02d}{7:02d}_{8:02d}{9:02d}_{10:s}.npz'.format(
            basedir,year,month,day,instrs[ind],year,month,day,hour,min,ch))
        files_exist = files_exist*path.isfile(files[-1])
        ind = ind+1
    print(files)
    if files_exist:
        if (not return_images):
            return files
        else:
            img = zeros(shape=(int(512/subsample),int(512/subsample),len(channels)),dtype='float32')
            for c in range(len(channels)):
                img[:,:,c] = sdo_scale(((load(files[c]))['x'])[::subsample,::subsample], channels[c])
            return img
    else:
        return -1

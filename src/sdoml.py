def sdo_read(year, month, day, hour, min, instr='AIA', channel='0094',subsample=1,basedir='/gpfs/gpfs_gl4_16mb/b9p111/b9p111ai/SDOML'):
    """
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

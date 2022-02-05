import numpy as np
from scipy.stats import pearsonr

def pixel2pixelcor(x, y):
    """ 
    compute correlation between two arrays. The output is a scalar.
    """
    x = np.ravel(x)
    y = np.ravel(y)
    corr = pearsonr(x, y)
    return corr
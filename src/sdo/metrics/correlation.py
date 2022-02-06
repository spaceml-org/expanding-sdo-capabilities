import numpy as np
from scipy.stats import pearsonr

def pixel2pixelcor(x, y):
    """ 
    compute correlation between two arrays. The output is a scalar.
    """
    
    x = x.flatten()
    y = y.flatten()
    corr = pearsonr(x, y)[0]
    return corr
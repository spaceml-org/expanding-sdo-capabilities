"""
This module collects functions to compute covariance between maps
"""
import numpy as np
from scipy.ndimage import uniform_filter, convolve


def neighbor_cov(X, Y, size=5):
    """
    Neighboorhood covariance with spatial mean on a rolling squared window.
    The output is a covariance map of the same size of X and Y. Each value
    in the map corresponds to the covariance on a squared patch centered 
    in the pixel and of dimension 'size'.
    cov_ij = sum_i_N [(x_i-x_mean)(y_i-y_mean)]/(N-1) where N=size**2
    
    X (np.array): first input
    Y (np.array): second input, expected ot have same
    shape as X
    size (int): size of rolling squared window over which average and sum
    are performed
    """
    X_mean = uniform_filter(X, size=size)
    Y_mean = uniform_filter(Y, size=size)
    A = X - X_mean
    B = Y - Y_mean
    C = A * B
    k = np.ones([size, size], dtype=int)
    den = X_mean.shape[0]*X_mean.shape[1]/(size -1)
    output = convolve(C, k, mode='constant', cval=0.0)/den
    return output


def time_covariance(X, Y):
    """
    Covariance between snapshots of two multi-dimensional arrays. 
    Average and sum are are over the snapshot index (i.e. time).
    The output is a covariance map of the same size of X and Y.
    
    X (np.ndarray): array of shape=(width_img,height_img, number of timestamps)
    Y (np.ndarray): same shape of X
    """
    meanX = np.mean(X, axis = 0)
    meanY = np.mean(Y, axis = 0)
    lenX = X.shape[0]
    X = X - meanX
    Y = Y - meanY
    cov = np.zeros_like(X)
    
    for i in range(lenX):
        cov += X[i,:,:]*Y[i,:,:]
    cov = sum(cov)/lenX
    print(f'The cov image has shape {cov.shape}')
    return cov

def cov_1d(x, y):
    """ 
    compute covariance between two arrays. The output is a scalar.
    """
    x = np.ravel(x)
    y = np.ravel(y)
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

    
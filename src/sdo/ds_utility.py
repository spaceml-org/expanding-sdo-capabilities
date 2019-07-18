"""
This module contains utils functions for statistics/data science on the data
"""
import numpy as np

def correlation_coefficient(patch1, patch2):
    """
    Compute the correlation between 2 patches.
    Parameters:
    patch1 (np.array): squared image
    patch2 (np.array): squared image, assumed to be of the same shape of patch1

    Return: float, coefficient of correlation
    """
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def compute_correlation(a, b, d=1):
    """
    compute the correlation coefficient between patches centered on the same pixel in the two images of interest

    Parameters:
    a (np.array): squared image
    b (np.array): squared image, assumed to be of the same shape of a
    d (int): width of the patch

    Return: float, coefficient of correlation
    """
    correlation = np.zeros_like(a)
    sh_row, sh_col = a.shape
    for i in range(d, sh_row - (d + 1)):
        upper = i - d
        bottom = i + d + 1
        for j in range(d, sh_col - (d + 1)):
            left = j - d
            right = j + d + 1
            correlation[i, j] = correlation_coefficient(
                a[upper:bottom, left:right], b[upper:bottom, left:right]
            )
    return correlation


def minmax_normalization(img):
    """
    Parameters:
    img (np.array): image to be normalized

    Returns: normalized image
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))
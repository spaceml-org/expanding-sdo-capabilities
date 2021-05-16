"""
This module contains functions for pipeline output visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_vt_sample(num_channels, input_data, output, gt_output, img_file, timestamp, index=0):
    """
    Given an index of the batch, it produces a figure that contains
    the input channels, the ground truth output channel and the reproduced
    channel. Index is the element of the batch that we want to plot.
    """
    # Print each of the input channels on their own line.
    fig = plt.figure()
    pos = 0
    timestamp = timestamp[index]
    for i in range(num_channels - 1):
        pos += 1

        ax = fig.add_subplot(num_channels + 1, 1, pos)
        fig.subplots_adjust(top=3.0)
        ax.set_title('Input Channel {} at time {}'.format((i + 1), timestamp))
        ax.axis('off')
        ax.imshow(input_data[index][i], vmin=0, vmax=1)
    pos += 1

    ax = fig.add_subplot(num_channels + 1, 1, pos)
    fig.subplots_adjust(top=3.0)
    ax.set_title('Predicted Output Channel {}'.format(num_channels))
    ax.axis('off')
    ax.imshow(output[index][0], vmin=0, vmax=1)
    pos += 1

    ax = fig.add_subplot(num_channels + 1, 1, pos)
    fig.subplots_adjust(top=3.0)
    ax.set_title('Ground Truth Output Channel {}'.format(num_channels))
    ax.axis('off')
    ax.imshow(gt_output[index][0], vmin=0, vmax=1)

    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def plot_2d_hist(img_gt, img_pred, img_file, timestamp, bins=100):
    """
    Given a predicted and a ground truth image, it plots a 2D histogram of the intensity. 
    The color scale represents the density of pixels.
    """
    plt.figure(figsize=(6, 6))
    H, xedges, yedges = np.histogram2d(img_gt.flatten(), img_pred.flatten(), bins=bins)
    plt.imshow(np.log10(H/1.), origin='lower',
               extent=(xedges.min(), 3, yedges.min(), 3),
               cmap='jet')
    plt.plot([xedges.min(), 3], [xedges.min(), 3])
    plt.colorbar()
    plt.title("2D density map at time {}".format(timestamp))
    plt.xlabel('Logarithm of the predicted intensity')
    plt.ylabel('Logarithm of the ground truth intensity')
    
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def plot_difference(img_gt, img_pred, img_file, timestamp, bins=5000):
    """
    The snippet below plots a 1d histrogram and a map of the difference between predicted 
    and ground truth image.
    """
    fig, axs = plt.subplots(1, 2, figsize=(
        10, 5), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    axs[0].hist((img_pred - img_gt).ravel(), density=True, bins=bins)
    axs[0].set_title('Histogram of the difference at time {}'.format(timestamp))
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlabel('Pred - GT Intensity')
    im1 = axs[1].imshow((img_pred - img_gt),
                        cmap='seismic', vmin=-1, vmax=1, origin='lower')
    axs[1].set_title('Pred - GT')
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()
    
    
def plot_2Dhist_95cl(Y_test, Y_pred, title='Model', mask_by_confidence=False, savefig=None, clower=-8, 
                     bins=200, xy_range=[[-2,3.0],[-2,3.0]], normed=False):
    """
    Given an array of predicttion and an array of ground truth values, it plots a 2D histogram of the intensity 
    at 95% confidence level. The color scale represents the density of pixels.
    """
    
    H, xedges, yedges = np.histogram2d(np.log10(Y_test.flatten()), 
                                       np.log10(Y_pred.flatten()), 
                                       bins=bins, range=xy_range, normed=normed)
 
    Hnorm = H/H.sum()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(title, fontsize=20)
    plt.plot(xy_range[0],xy_range[1], linewidth=3)
    divider = make_axes_locatable(ax)
    im = ax.imshow(np.log10(Hnorm.T+1e-20), origin='lower', clim=(clower,0), 
               extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()),
               cmap='jet', aspect='equal', interpolation='none')
    cax = divider.append_axes('right', size='5%', pad=0.07)
    ax.set_xlabel('Log10 Real Intensity',fontsize=20)
    ax.set_ylabel('Log10 Predicted Intensity', fontsize=20)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    Hcum = Hnorm.copy()
    for i in range(Hnorm.shape[1]):
        Hcum[:,i]=H[:,i].cumsum(axis=0)/H[:,i].sum()
    Hcum[np.where(np.isnan(Hcum))] = 1.0
    mask = np.ones(Hcum.shape)
    mask[np.where(np.abs(Hcum-0.5) <= 0.45)] = np.nan
    if mask_by_confidence:
        ax.imshow(mask.T,extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()),
               origin='lower', cmap='binary', alpha=1.0)
    plt.tick_params(axis='both', which='major', size=15)
    if savefig == None:
        plt.show()
    else:
        plt.savefig(savefig)

"""
In this module we collect functions to plot comparison of models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pred_and_gt(results_path: str, revert_root: bool = False) -> (np.array, np.array):
    """
    Load predictions and ground truths from file, optionally remove
    root scaling, sort values and removes negative gt pixels.
    Args:
        results_path: path to file containing gt and predictions in npz format
        revert_root: if True both predictions and ground truth are **2

    Returns:
        Y_test, Y_pred
    """
    Y = np.load(results_path)
    shape = Y.shape
    Y_test = Y[:, :, 0:int(shape[2] / 2), :]
    Y_pred = Y[:, :, int(shape[2] / 2):, :]

    if revert_root:
        Y_test = np.power(Y_test, 2)
        Y_pred = np.power(Y_pred, 2)

    Y_test = Y_test.flatten()
    Y_pred = Y_pred.flatten()
    idx = np.argsort(Y_test)
    Y_test = Y_test[idx]
    Y_pred = Y_pred[idx]

    mask1 = Y_test > 0
    Y_test = Y_test[mask1]
    Y_pred = Y_pred[mask1]

    return Y_test, Y_pred


def compute_hist_values(Y_test, bins=50, xrange=(-4, 2)):
    y, binEdges = np.histogram(Y_test, bins=bins, range=xrange)
    ynorm = y / len(Y_test) * 100
    binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    return ynorm, binEdges, binCenters


def datapoints_to_bins(Y_test, binEdges, binCenters):
    #TODO parallelize, too slow this way
    i = 0
    l_bins = []
    l_bincenters = []
    for val in Y_test:
        if val <= binEdges[i + 1]:
            l_bins.append(i)
        else:
            i = i + 1
            l_bins.append(i)
        l_bincenters.append(binCenters[i])
    return l_bins, l_bincenters


def create_df_errors(Y_test, Y_pred):
    log10Y_test = np.log10(Y_test)
    log10Y_pred = np.log10(Y_pred)

    #by design we split the values in log scale
    ynorm, binEdges, binCenters = compute_hist_values(log10Y_test, bins=50, xrange=(-4, 2))
    l_bins, l_bincenters = datapoints_to_bins(log10Y_test, binEdges, binCenters)

    df = pd.DataFrame(
        {'Bins': l_bins,
         'BinCenters': l_bincenters,
         'YTest': Y_test,
         'YPred': Y_pred,
         'log10_YTest': log10Y_test,
         'log10_Ypred': log10Y_pred,
         }
    )
    df['YTest-YPred'] = df.YTest - df.YPred
    df['log10_YTest-log10_YPred'] = df.log10YTest - df.log10YPred
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['(YTest-YPred)/YTest'] = (df.YTest - df.YPred) / df.YTest * 100
    return df


def create_df_quantiles(df, frac_sample, quantiles, groupby_col, val_col):
    df_q = df[[groupby_col, val_col]].\
        sample(frac=frac_sample).groupby(groupby_col).\
        quantile(quantiles).unstack()
    df_q = df_q.reset_index()
    return df_q


def create_df_combined_plots(Y_test, Y_pred, frac_sample=1.0, quantiles=[0.05, 0.5, 0.95],
                             groupby_col='BinCenters', val_col='log10_YTest-log10_YPred'):
    df = create_df_errors(Y_test, Y_pred)
    df_q = create_df_quantiles(df, frac_sample, quantiles, groupby_col, val_col)
    return df, df_q


def create_combined_plots(exp_name, df1_q, df2_q, groupby_col, val_col, hbin_centers, hynorm,
                          label1='root scaling', label2='no scaling', hwidth=0.1):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle('Error on Predictions by True Intensity - 95% c.l.')

    ax1.plot(df1_q[groupby_col], df1_q[val_col][0.5], color='blue', label='median_' + label1)
    ax1.plot(df2_q[groupby_col], df2_q[val_col][0.5], color='green', label='median_' + label2)
    ax1.fill_between(df1_q[groupby_col], df1_q[val_col][0.05],
                     df1_q[val_col][0.95],
                     label=label1, color='blue', alpha=0.2)
    ax1.fill_between(df2_q[groupby_col], df2_q[val_col][0.05],
                     df2_q[val_col][0.95],
                     label=label2, alpha=0.2, color='green')
    ax1.plot([-4, 2], [0, 0])
    ax1.legend()
    ax1.set_ylabel(val_col)

    ax2.bar(hbin_centers, hynorm, width=hwidth, color='b', alpha=0.8)
    ax2.set_xlabel('Log10 Real Intensity')
    ax2.set_ylabel('% Pixels')
    ax2.set_xlim(-4, 2)
    ax1.grid()
    ax2.grid()

    fig_title = exp_name + 'error_hist.pdf'
    plt.savefig(fig_title, bbox_inches='tight')
    plt.show()




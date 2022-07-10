"""
In this module we collect functions to plot comparison of models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import logging
import math


def load_pred_and_gt(results_path: str, revert_root: bool = False, 
                     frac: Optional[float]=1.0) -> Tuple[np.array, np.array]:
    """
    Load predictions and ground truths from file, optionally remove
    root scaling, sort values and removes negative gt pixels.
    Args:
        results_path: path to file containing gt and predictions in npz format
        revert_root: if True both predictions and ground truth are **2
        frac: percentage of datapoints to be loaded, selected randomly, this is 
            particularly helpful for testing purposes

    Returns:
        Y_test, Y_pred
    """
    Y = np.load(results_path)
    shape = Y.shape
    Y_test = Y[:, :, 0:int(shape[2] / 2), :]
    Y_pred = Y[:, :, int(shape[2] / 2):, :]

    if revert_root:
        logging.info('Reverting root scaling')
        Y_test = np.power(Y_test, 2)
        Y_pred = np.power(Y_pred, 2)

    Y_test = Y_test.flatten()
    Y_pred = Y_pred.flatten()
    if frac< 1.0:
        Y_size = Y_test.size
        sample_size = int(Y_size * frac)
        random_idx = np.random.choice(np.arange(Y_size), size=sample_size, replace=False)
        Y_test = Y_test[random_idx]
        Y_pred = Y_pred[random_idx]
        
    idx = np.argsort(Y_test)
    Y_test = Y_test[idx]
    Y_pred = Y_pred[idx]

    mask1 = Y_test > 0
    Y_test = Y_test[mask1]
    Y_pred = Y_pred[mask1]

    return Y_test, Y_pred


def compute_hist_values(Y_test: np.array, bins: Optional[int] = 50,
                        xrange: Optional[Tuple[int, int]] = (-4, 3)) -> Tuple[np.array, np.array, np.array]:
    y, binEdges = np.histogram(Y_test, bins=bins, range=xrange)
    ynorm = y / len(Y_test) * 100
    binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    return ynorm, binEdges, binCenters


def datapoints_to_bins(Y_test: np.array, binEdges: np.array, binCenters: np.array) -> Tuple[List[int], List[float]]:
    #TODO parallelize, too slow this way
    j = 0
    l_bins = []
    l_bincenters = []
    print('min Y test', min(Y_test))
    print('min bin Edges', min(binEdges))
    print('max Y test', max(Y_test))
    print('max bin Edges', max(binEdges))
    for i in range(len(Y_test)):
        if Y_test[i] <= binEdges[j + 1]:
            l_bins.append(j)
        else:
            j=j+1
            l_bins.append(j)
            if j > 49:
                import pdb;pdb.set_trace()
        l_bincenters.append(binCenters[j])
        i = i + 1
    return l_bins, l_bincenters


def create_df_errors(Y_test: np.array, Y_pred: np.array, bins: Optional[int] = 50,
                        xrange: Optional[Tuple[int, int]] = (-4, 2)) -> pd.DataFrame:
    log10Y_test = np.log10(Y_test)
    log10Y_pred = np.log10(Y_pred)

    #by design we split the values in log scale
    ynorm, binEdges, binCenters = compute_hist_values(log10Y_test, bins=bins, xrange=xrange)
    l_bins, l_bincenters = datapoints_to_bins(log10Y_test, binEdges, binCenters)

    df = pd.DataFrame(
        {'Bins': l_bins,
         'BinCenters': l_bincenters,
         'YTest': Y_test,
         'YPred': Y_pred,
         'log(YTest)': log10Y_test,
         'log(YPred)': log10Y_pred,
         }
    )
    df['YTest-YPred'] = df.YTest - df.YPred
    df['log(YTest)-log(YPred)'] = df['log(YTest)'] - df['log(YPred)']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['(YTest-YPred)/YTest'] = (df.YTest - df.YPred) / df.YTest * 100
    return df


def create_df_quantiles(df: pd.DataFrame, frac_sample: float, quantiles: List[float],
                        groupby_col: str, val_col: str) -> pd.DataFrame:
    df_q = df[[groupby_col, val_col]].\
        sample(frac=frac_sample).groupby(groupby_col).\
        quantile(quantiles).unstack()
    df_q = df_q.reset_index()
    return df_q


def create_df_combined_plots(Y_test : np.array, Y_pred: np.array, bins: Optional[int] = 50,
                             xrange: Optional[Tuple[int, int]] = (-4, 2), frac_sample: Optional[float] = 1.0,
                             quantiles: Optional[List[int]] = [0.05, 0.5, 0.95],
                             groupby_col: Optional[str] = 'BinCenters',
                             val_col: Optional[str] = 'log(YTest)-log(YPred)') -> (pd.DataFrame, pd.DataFrame):
    df = create_df_errors(Y_test, Y_pred, bins, xrange)
    df_q = create_df_quantiles(df, frac_sample, quantiles, groupby_col, val_col)
    return df, df_q


def create_combined_plots(exp_name: str, output_path: str, df1: pd.DataFrame,df1_q: pd.DataFrame, 
                          df2_q: pd.DataFrame,
                          groupby_col: Optional[str] = 'BinCenters', 
                          val_col: Optional[str] = 'log(YTest)-log(YPred)',
                          label1: Optional[str] = 'root',
                          label2: Optional[str] = 'no_root',
                          hwidth: Optional[float] = 0.1, hcol: Optional[str] = 'log(YTest)',
                          hbins: Optional[int] = 50, xrange: Optional[Tuple[int, int]] = (-4, 2)):
   
    scaling_factors = {'94': 10, '171': 2000, '193': 3000, '211': 1000, }
    alpha = scaling_factors[exp_name]
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 8))
    fig.suptitle(exp_name, size=24)
    fig.subplots_adjust(hspace=0, wspace=0)
    #fig.suptitle('Error on Predictions by True Intensity - 95% c.l.')

    ax1.plot(df1_q[groupby_col], df1_q[val_col][0.5], color='blue', label='median ' + label1)
    ax1.plot(df2_q[groupby_col], df2_q[val_col][0.5], color='green', label='median ' + label2)
    ax1.fill_between(df1_q[groupby_col], df1_q[val_col][0.05],
                     df1_q[val_col][0.95],
                     label='95% c.l. '+ label1, color='blue', alpha=0.2)
    ax1.fill_between(df2_q[groupby_col], df2_q[val_col][0.05],
                     df2_q[val_col][0.95],
                     label='95% c.l. '+ label2, alpha=0.2, color='green')
    ax1.plot(xrange, [0, 0])
    ax1.legend()
    ax1.set_ylim([-2.5, 1])
    ax1.set_ylabel(val_col, fontsize=18)
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    ax1.grid()
    
    ynorm, _, bin_centers = compute_hist_values(df1[hcol], bins=hbins, xrange=xrange)
    ax2.bar(bin_centers, ynorm, width=hwidth, color='b', alpha=0.8)
    ax2.set_xlabel('Log(True Count Rate)', fontsize=18)
    ax2.set_ylabel('% Pixels', fontsize=18)
    ax2.set_xlim(xrange)
    ax2.set_ylim([0, 13])
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_yticklabels(), fontsize=14)
    ax2.grid()
    

    fig_title = output_path + exp_name + 'error_hist.pdf'
    plt.savefig(fig_title, bbox_inches='tight')
    plt.show()

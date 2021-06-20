"""
In this module we collect functions to plot comparison of models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


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


def compute_hist_values(Y_test: np.array, bins: Optional[int] = 50,
                        xrange: Optional[tuple[int, int]] = (-4, 2)) -> (np.array, np.array, np.array):
    y, binEdges = np.histogram(Y_test, bins=bins, range=xrange)
    ynorm = y / len(Y_test) * 100
    binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    return ynorm, binEdges, binCenters


def datapoints_to_bins(Y_test: np.array, binEdges: np.array, binCenters: np.array) -> (list[int], list[float]):
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


def create_df_errors(Y_test: np.array, Y_pred: np.array, bins: Optional[int] = 50,
                        xrange: Optional[tuple[int, int]] = (-4, 2)) -> pd.DataFrame:
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


def create_df_quantiles(df: pd.DataFrame, frac_sample: float, quantiles: list[float],
                        groupby_col: str, val_col: str) -> pd.DataFame:
    df_q = df[[groupby_col, val_col]].\
        sample(frac=frac_sample).groupby(groupby_col).\
        quantile(quantiles).unstack()
    df_q = df_q.reset_index()
    return df_q


def create_df_combined_plots(Y_test : np.array, Y_pred: np.array, frac_sample: Optional[float] = 1.0,
                             quantiles: Optional[list[int]] = [0.05, 0.5, 0.95],
                             groupby_col: Optional[str] = 'BinCenters',
                             val_col: Optional[str] = 'log10_YTest-log10_YPred') -> (pd.DataFame, pd.DataFame):
    df, ynorm, binCenters = create_df_errors(Y_test, Y_pred)
    df_q = create_df_quantiles(df, frac_sample, quantiles, groupby_col, val_col)
    return df, df_q


def create_combined_plots(exp_name: str, output_path: str, df1_q: pd.DataFame, df2_q: pd.DataFame,
                          groupby_col: str, val_col: str, label1: Optional[str] = 'root scaling',
                          label2: Optional[str] = 'no scaling',
                          hwidth: Optional[float] = 0.1, hcol: Optional[str] = 'log10_YTest',
                          hbins: Optional[int] = 50, xrange: Optional[tuple[int, int]] = (-4, 2)):

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

    ynorm, _, bin_centers = compute_hist_values(df1_q[hcol], bins=hbins, xrange=xrange)
    ax2.bar(bin_centers, ynorm, width=hwidth, color='b', alpha=0.8)
    ax2.set_xlabel('Log10 Real Intensity')
    ax2.set_ylabel('% Pixels')
    ax2.set_xlim(-4, 2)
    ax1.grid()
    ax2.grid()

    fig_title = output_path + exp_name + 'error_hist.pdf'
    plt.savefig(fig_title, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    data_inventory = '/home/Valentina/inventory_1904.pkl'
    results_path = '/fdl_sdo_data/bucket/EXPERIMENT_RESULTS/VIRTUAL_TELESCOPE'
    output_path = '/home/Valentina/'
    pred1_path = results_path + '/vale_exp_23/0600_vale_exp_23_train_predictions.npy'
    pred2_path = results_path + '/vale_exp_20/0600_vale_exp_20_train_predictions.npy'
    Y1_test, Y1_pred = load_pred_and_gt(pred1_path, revert_root=True)
    Y2_test, Y2_pred = load_pred_and_gt(pred2_path, revert_root=True)
    df1, df1_q = create_df_combined_plots(Y1_test, Y1_pred, val_col = 'log10_YTest-log10_YPred')
    df2, df2_q = create_df_combined_plots(Y2_test, Y2_pred, val_col='log10_YTest-log10_YPred')
    create_combined_plots('211_rootscaling', output_path, df1_q, df2_q, val_col= 'log10_YTest-log10_YPred',
                          label1 = 'root scaling', label2 = 'no scaling')



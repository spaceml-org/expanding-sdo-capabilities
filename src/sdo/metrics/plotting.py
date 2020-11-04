"""
In this module we collect functions useful to produce some metric visualizations
"""
import os
import logging

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from sdo.io import format_graph_prefix

_logger = logging.getLogger(__name__)


def plot_regression(predicted, ground_truth, title, x_label, y_label, filename):
    """
    This produces a joint plot that contains a regression plot and histogram of the
    variables. It also contains the Pearson value.
    
    predicted (np.array)
    ground_truth (np.array)
    title (str) 
    x_label (str)
    y_label (str)
    filename (str)
    """
    sns.set(style="white", color_codes=True)
    ax = sns.jointplot(x=predicted, y=ground_truth, kind='reg')
    ax.set_axis_labels(x_label, y_label)
    plt.title(title)
    plt.tight_layout()
    ax.annotate(stats.pearsonr)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_loss(epoch, train_losses, test_losses, results_path, exp_name):
    """
    Plot both training and testing losses on the same graph.
    """
    fig = plt.figure()

    # Prevent large unscaled ReLU values at the very start of training
    # from 'dominating' on the loss y-axis.
    if len(train_losses) > 20:
        x=20
    else:
        x=0
    if min(train_losses) > 0:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.gca().set_ylim([min(train_losses), 
                            max(test_losses[x:])]
                          )
    else:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.gca().set_ylim([min(train_losses[x:]), 
                            max(test_losses)])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/testing loss after {} epochs'.format(epoch))
    img_file = os.path.join(results_path, '{}_loss_graph.png'.format(
        format_graph_prefix(epoch, exp_name)))
    plt.legend()
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()
    _logger.info('\nTraining/testing loss graph for epoch {} saved to {}'.format(
        epoch, img_file))


def plot_primary_metric(epoch, train_primary_metrics, test_primary_metrics,
                        results_path, exp_name, metric_name):
    """
    Plot both training and testing primary metrics on the same graph.
    """
    fig = plt.figure()
    plt.plot(train_primary_metrics, label='Training Primary Metric')
    plt.plot(test_primary_metrics, label='Testing Primary Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Primary metric')
    plt.title(
        'Training/testing primary metric ({}) after {} epochs'.format(
            metric_name, epoch))
    img_file = os.path.join(results_path, '{}_primary_metric_graph.png'.format(
        format_graph_prefix(epoch, exp_name)))
    plt.legend()
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()
    _logger.info('Training/testing primary metric graph for epoch {} saved to {}'.format(
        epoch, img_file))
"""
In this module we collect functions useful to produce some metric visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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

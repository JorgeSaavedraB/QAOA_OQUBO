import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def error_line_graph(ax, x, raw_data, labels, title, x_label, y_label, markers=None, lss=None, palette='bright', loc="center right"):
    """
    Plot line graphs with mean ± SEM shading using Seaborn styling.

    Parameters
    ----------
    x : list of arrays
        List of x-values for each line (same length as raw_data)
    raw_data : list of 2D arrays
        Each element is an array (subjects/trials × timepoints)
    labels : list of str
        Labels for each line
    title : str
        Plot title
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    y_lim : tuple or None
        y-axis limits (optional)
    palette : str or list
        Seaborn color palette name or custom list of colors
    """

    sns.set(style="ticks", rc={"axes.linewidth": 1.5, 'xtick.major.width':1.5, 'ytick.major.width':1.5, 'xtick.labelsize': 15, 'ytick.labelsize': 15}, context="talk", palette=palette)
    colors = sns.color_palette(palette, n_colors=len(raw_data))

    if markers==None:
        markers=['.'] * len(labels)
    if lss==None:
        lss=['-'] * len(labels)

    for i, (x_vals, y_data, label, marker, ls) in enumerate(zip(x, raw_data, labels, markers, lss)):
        y_data = np.array(y_data)
        
        # Compute mean and SEM
        y_mean = np.mean(y_data, axis=1)
        y_sem = np.std(y_data, axis=1, ddof=1) / np.sqrt(y_data.shape[1])

        ax.errorbar(x_vals, y_mean, yerr=y_sem, ls=ls, lw=1.5, marker=marker, label=label, capsize=3, capthick=1.5)
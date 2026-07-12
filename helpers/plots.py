


import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


from helpers.outils import MaC



def plot_rectangular_MAC(MAC, language='Spanish', modes_number=None,
               annot_kws=None, annot_kws_ticklabels=None,
               annot_kws_legend=None, vmin=10**-5, vmax=1):
    """
    modes_number: list for the labels.
        Example: if we want to label "Mode 1, Mode  4, Mode 7" then modes_number=[1, 4, 7]
    Example for annot_kws: annot_kws = {"fontsize": 10, "fontweight": "bold", "color": "black"}
    """

    n_rows, n_cols = np.shape(MAC)

    if modes_number is not None and len(modes_number) == 2:
        modes_rows, modes_cols = modes_number
    else:
        modes_rows = list(range(1, n_rows + 1))
        modes_cols = list(range(1, n_cols + 1))

    if language == 'Spanish':
        label = 'Modo'
    else:
        label = 'Mode'

    rows = [f"{label} {i}" for i in modes_rows]
    cols = [f"{label} {j}" for j in modes_cols]

    MAC = pd.DataFrame(MAC, index=rows, columns=cols)
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(MAC, cmap="jet", ax=ax, annot=True,
                fmt='.3f', annot_kws=annot_kws, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if annot_kws_ticklabels is not None:
        if 'fontsize' in annot_kws_ticklabels:
            for tick_label in ax.get_xticklabels():
                tick_label.set_fontsize(annot_kws_ticklabels["fontsize"])
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontsize(annot_kws_ticklabels["fontsize"])
        if 'fontweight' in annot_kws_ticklabels:
            for tick_label in ax.get_xticklabels():
                tick_label.set_fontweight(annot_kws_ticklabels["fontweight"])
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontweight(annot_kws_ticklabels["fontweight"])
        # Add other properties as needed

    if annot_kws_legend is not None:
        cbar = heatmap.collections[0].colorbar
        for t in cbar.ax.get_yticklabels():
            if 'fontsize' in annot_kws_legend:
                t.set_fontsize(annot_kws_legend["fontsize"])
            if 'fontweight' in annot_kws_legend:
                t.set_fontweight(annot_kws_legend["fontweight"])

    fig.tight_layout()

    return (fig, ax)

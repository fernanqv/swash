import os
import os.path as op
import sys

import numpy as np
import pandas as pd
import glob
#from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

props = dict(boxstyle='round', facecolor='w', edgecolor='grey', linewidth=0.8, alpha=0.5)

# --------------------------------------------------------------------------------
# diagnostic statistics
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# relative
def rmse_relative(predictions, targets):
    return np.sqrt((((predictions - targets)/targets) ** 2).mean())

def bias(predictions, targets):
    return sum(predictions-targets)/len(predictions)

def si(predictions, targets):
    S = predictions.mean()
    O = targets.mean()
    return np.sqrt(sum(((predictions-S)-(targets-O))**2)/((sum(targets**2))))

def gauss_kde(x, y):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    return(z)

# --------------------------------------------------------------------------------

def scatter_color(dataset, target, vars_dataset, var_target_color, figsize, vmin, vmax, cmap):
    
    # scatterplot every variable (using matplotlib.gridspec)
    vnames = dataset.columns
    fig, axs = plt.subplots(
        len(vars_dataset)-1, len(vars_dataset)-1, 
        figsize=figsize,
        sharex=False, sharey=False,
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for c1, v1 in enumerate(vars_dataset[1:]):
        for c2, v2 in enumerate(vars_dataset[:-1]):
            
            im = axs[c2,c1].scatter(dataset[v1], dataset[v2], marker=".", c=target[var_target_color].values, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.6)
            axs[c2,c1].set_facecolor('whitesmoke')
            axs[c2,c1].patch.set_alpha(0.7)
            axs[c2,c1].grid(c='w', linewidth=1.4)
            axs[c2,c1].set_axisbelow(True)
            if c1==c2:
                axs[c2,c1].set_xlabel(vars_dataset[c1+1])
                axs[c2,c1].set_ylabel(vars_dataset[c2])
            elif c1>c2:
                axs[c2,c1].xaxis.set_ticklabels([])
                axs[c2,c1].yaxis.set_ticklabels([])
                
            else:
                fig.delaxes(axs[c2, c1])
    
    fig.colorbar(im, ax=axs.ravel().tolist())
    return(fig)

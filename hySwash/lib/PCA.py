import os
import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

from .RBF import RBF_Reconstruction
from .MDA import scatter_mda

def PCs(df, variance, norm=False):
    '''
    Princial Componenet Analysis
    '''
    # Normalized DataFrame
    
    df = df.dropna(axis=1)
    df_mean = np.nanmean(df, axis=0)
    df_std = np.std(df, axis=0)
    
    if norm:
        df_norm = (df - df_mean) / df_std
    
    else:
        df_norm = df
    
    # principal components analysis
    ipca = PCA(n_components=min(df_norm.shape[0], df_norm.shape[1]))
    PCs = ipca.fit_transform(df_norm)

    # store PCA output in a xarray.Dataset
    xds_PCA = xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),
        }
    )
    
    # PCs explaining x % of variance
    var_tot = np.sum(xds_PCA.variance.values)
    n_percent = xds_PCA.variance * 100 / var_tot
    num = np.where(np.cumsum(n_percent) >= variance)[0]
    xds_Var = xds_PCA.sel(n_components=slice(0, num[0]+1))

    return(xds_Var, df_mean, df_std, var_tot)


def k_iterator(var, sub_splt, df_input, df_output, ix_scalar_subset, ix_directional_subset, variance):
    'Iterate along the dataframe subdivisions. Calculate PCs and Interpolate RBF'
    
    array_list = []
    
    # k-iterator
    RMSE = []
    for pseg, seg_recons in enumerate(sub_splt):

        # index diff
        ix_comp = set(seg_recons.index.values).symmetric_difference(set(df_input.index.values))

        # select data used 
        seg_subset = df_input.loc[ix_comp]
        seg_target = df_output.loc[ix_comp]
        
        # calculate PCs
        xds_PCA, mean, std, var_tot = PCs(seg_target, variance)
        
        # define indexes scalar and directional
        ix_scalar_target = range(len(xds_PCA.n_components))
        ix_directional_target = [] 

        # RBF 
        target_PCs = pd.DataFrame(xds_PCA['PCs'].values)
        output = RBF_Reconstruction(
            seg_subset.values[:, ix_scalar_subset], ix_scalar_subset, ix_directional_subset,
            target_PCs.values[:, ix_scalar_target], ix_scalar_target, ix_directional_target,
            seg_recons.values[:, ix_scalar_subset]
        )

        # Storage RBF output to xr.Dataset
        ds_output = xr.Dataset({'PCs': (('case', 'n_components'), output)},
               coords = {
                   'case': seg_recons.index.values,
                   'n_components': xds_PCA.n_components
                        }
              )
        ds_output = ds_output.sortby('case')

        dataset = ((ds_output['PCs'] * xds_PCA['EOFs']).sum(dim='n_components') + mean).to_dataset(name=var)
        
        #dataset[var] = dataset[var] * std.values + mean

        # PCs & EOFs reconstruction
        
        array_list.append(dataset)

    dataset_output = xr.concat(array_list, dim='case')
    
    return (dataset_output, xds_PCA, mean, std)

def plot_PCA(xds_PCA, dataset, dims, figsize1, figsize2):
    '''
    input:
        n_components: number of PCs

    '''
    # plot PCs and EOFs
    for pc in xds_PCA['n_components'].values:

        # plot EOFs values over subset sample
        xds = xds_PCA.sel(n_components=pc)
        EOF = xds['EOFs']
        var = xds['variance'].values

        vr = np.nanmax([np.abs(EOF.min()), np.abs(EOF.max())]) 

        fig, ax = plt.subplots(1, figsize=figsize2)
        ax.plot(EOF['Xp'], EOF.values, c='grey')
        ax.scatter(EOF['Xp'], EOF.values, c=EOF.values, vmin=-vr, vmax=vr, cmap='bwr')

        # ax config
        ax.set_title('EOF {0} - {1:.2f}% variance explained'.format(pc, var))
        ax.set_xlim(xds_PCA['Xp'].values.min(), xds_PCA['Xp'].values.max())
        ax.set_facecolor('whitesmoke')
        ax.patch.set_alpha(0.7)
        ax.grid(c='w', linewidth=1.4)
        ax.set_ylabel('PC')
        ax.set_xlabel('X')

        # plot PCs values over subset sample
        fig = scatter_mda([dataset[dims]], names = dims, color=[xds_PCA.isel(n_components=pc)['PCs']], figsize=figsize1, ss=[40])

        plt.show()
        
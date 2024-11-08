#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import os.path as op
import sys

# addpath to bluemath modules
sys.path.insert(0, op.join(os.getcwd(), '..', '..'))

from .RBF import *
from .PCA import *


def RBF_Reconstruction_singular(
    sdp, var, dims, df_dataset, df_subset,
    ix_scalar_subset, ix_directional_subset, 
    ix_scalar_target, ix_directional_target):
     
    'RBF'

    ds_target = sdp.ds_output[var]
    df_target = ds_target.to_dataframe()

    output = RBF_Reconstruction(
        df_subset[dims].values, ix_scalar_subset, ix_directional_subset,
        df_target[var].values, ix_scalar_target, ix_directional_target,
        df_dataset[dims].values)

    df_output = pd.DataFrame(data=output, index=df_dataset.index, columns=var)

    return df_output


def RBF_Reconstruction_spatial(
    sdp, var, dims, df_dataset, df_subset,
    ix_scalar_subset, ix_directional_subset, 
    ix_scalar_target, ix_directional_target, variance, X_max):

    'PCA + RBF'

    # take variable in dataset output
    ds_target = sdp.ds_output[var]

    # limit maximum Xp to avoid noise close to the beach
    ds_target = ds_target.sel(Xp=slice(None, X_max))

    # convert dataset to dataframe
    df_target = ds_target.to_dataframe()
    
    df_target = df_target.reset_index()[['case_id', 'Xp', var[0]]]
    
    # pivot dataframe
    df_target = df_target.pivot(index='case_id', columns='Xp', values=var[0])

    # compute PCA
    xds_PCA, mean, std, var_tot = PCs(df_target, variance, norm=False)

    # convert variance to percentage
    xds_PCA['variance'] = (xds_PCA['variance'] / var_tot)*100

    # define indexes scalar and directional from PC-target
    ix_scalar_target = range(len(xds_PCA.n_components))
    ix_directional_target = [] 
    
    # RBF 
    target_PCs = pd.DataFrame(xds_PCA['PCs'].values)
    output = RBF_Reconstruction(
            df_subset[dims].values, ix_scalar_subset, ix_directional_subset,
            target_PCs.values, ix_scalar_target, ix_directional_target,
            df_dataset[dims].values
        )

    # Storage RBF output to xr.Dataset
    ds_output_RBF = xr.Dataset({'PCs': (('case', 'n_components'), output)},
            coords = {
                'case': df_dataset.index.values,
                'n_components': xds_PCA.n_components
                    }
        )
    ds_output_RBF = ds_output_RBF.sortby('case')
    
    da_mean = xr.DataArray(data=mean, dims=['Xp'], coords={'Xp': ds_target['Xp'].values})
    xds_PCA = xds_PCA.rename({'n_features': 'Xp'})
    xds_PCA['Xp'] = ds_target['Xp'].values
    
    ds_output = ((ds_output_RBF['PCs'] * xds_PCA['EOFs']).sum(dim='n_components') + da_mean).to_dataset(name=var[0])
    
    return xds_PCA, ds_output

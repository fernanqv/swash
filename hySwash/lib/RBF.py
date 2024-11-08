#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import fminbound
from scipy import stats

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from .MDA import Normalize

# RBF Phi functions
def rbfphi_linear(r, const):
    return r

def rbfphi_cubic(r, const):
    return r*r*r

def rbfphi_gaussian(r, const):
    return np.exp(-0.5*r*r/(const*const))

def rbfphi_multiquadratic(r, const):
    return np.sqrt(1+r*r/(const*const))

def rbfphi_thinplate(r, const):
    return r*r*log(r+1)


def RBF_Assemble(x, phi, const, smooth):

    dim, n = x.shape
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            r = np.linalg.norm(x[:,i]-x[:,j])
            temp = phi(r, const)
            A[i,j] = temp
            A[j,i] = temp
        A[i,i] = A[i,i] - smooth

    # polynomial part
    P = np.hstack((np.ones((n,1)), x.T))
    A = np.vstack(
        (np.hstack((A,P)), np.hstack((P.T, np.zeros((dim+1,dim+1)))))
    )

    return A

def CostEps(ep, x, y):

    # rbf coeff and A matrix calculation
    rbf_coeff, A = CalcRBF_Coeff(ep, x, y)

    # rbf cost calculation
    m, n = x.shape
    A = A[:n,:n]
    invA = np.linalg.pinv(A)
    m1, n1 = rbf_coeff.shape
    kk = y - rbf_coeff[m1-m-1]
    for i in range(m):
        kk = kk - rbf_coeff[m1-m+i] * x[i,:]
    ceps = np.dot(invA, kk) / np.diagonal(invA)
    yy = np.linalg.norm(ceps)

    return yy

def CalcRBF_Coeff(ep, x, y):

    # rbf coeff calculation
    m, n = x.shape
    A = RBF_Assemble(x, rbfphi_gaussian, ep, 0)
    b = np.concatenate((y, np.zeros((m+1,)))).reshape(-1,1)
    rbfcoeff, _, _ ,_ = np.linalg.lstsq(A, b, rcond=None)  # inverse

    return rbfcoeff, A

def RBF_Interpolation(rbf_constant, rbf_coeff, nodes, x):

    phi = rbfphi_gaussian   # gaussian RBFs
    rbf_coeff = rbf_coeff.flatten()

    dim, n = nodes.shape
    dim_p, n_p = x.shape

    f = np.zeros(n_p)
    r = np.zeros(n)

    for i in range(n_p):
        r = np.linalg.norm(
            np.repeat([x[:,i]], n, axis=0)-nodes.T,
            axis=1
        )
        s = rbf_coeff[n] + np.sum(rbf_coeff[:n] * phi(r, rbf_constant))

        # linear part
        for k in range(dim):
            s = s + rbf_coeff[k+n+1] * x[k,i]

        f[i] = s

    return f

def RBF_Calibration(
    subset, ix_scalar_subset, ix_directional_subset,
    target, ix_scalar_target, ix_directional_target,
    ):
    '''
    Radial Basis Function (Gaussian) calibration.

    subset                - subset used for fitting RBF (dim_input)
    ix_scalar_subset      - scalar columns indexes for subset
    ix_directional_subset - directional columns indexes for subset
    target                - target used for fitting RBF (dim_output)
    ix_scalar_target      - scalar columns indexes for target
    ix_directional_target - directional columns indexes for target
    '''
    # parameters
    sigma_min = 0.001
    sigma_max = 0.1
    
    subset_norm, mins, maxs = Normalize(
        subset, ix_scalar_subset, ix_directional_subset)
    
    d_sigma = 0
    
    # RBF scalar variables 
    rbf_coeffs, opt_sigmas = [], []
    output_scalar, output_dir_x, output_dir_y = [], [], []
    
    for ix in ix_scalar_target:
        v = target[:,ix]
        
        # minimize RBF cost function
        t0 = time.time()  # time counter
        
        # ensure that sigma opt is in the bounds of [sigma_min, sigma_max]
        while d_sigma < 0.0001:
            opt_sigma = fminbound(
                CostEps, sigma_min, sigma_max, args=(subset_norm.T, v)
            )
            lm_min = np.abs(opt_sigma-sigma_min)
            lm_max = np.abs(opt_sigma-sigma_max)
            
            if lm_min < 0.0001:
                sigma_min = sigma_min - sigma_min/2
                
            elif lm_max < 0.001:
                sigma_max = sigma_max + sigma_max/2
            
            d_sigma = np.nanmin([lm_min, lm_max])
            
        t1 = time.time()  # optimization time
        
        # calculate RBF coeff
        rbf_coeff, _ = CalcRBF_Coeff(opt_sigma, subset_norm.T, v)
        output_scalar.append(np.reshape(rbf_coeff, -1))
        
        #rbf_coeffs.append(rbf_coeff)
        opt_sigmas.append(opt_sigma)
    
    # RBF directional variables
    for ix in ix_directional_target:
        v = target[:,ix]

        # x and y directional variable components
        vdg = np.pi/2 - v * np.pi/180
        pos = np.where(vdg < -np.pi)[0]
        vdg[pos] = vdg[pos] + 2 * np.pi
        vdx = np.cos(vdg)
        vdy = np.sin(vdg)

        # minimize RBF cost function
        t0 = time.time()  # time counter
        opt_sigma_x = fminbound(
            CostEps, sigma_min, sigma_max, args=(subset_norm.T, vdx)
        )
        opt_sigma_y = fminbound(
            CostEps, sigma_min, sigma_max, args=(subset_norm.T, vdy)
        )
        t1 = time.time()  # optimization time

        # calculate RBF coeff
        rbf_coeff_x, _ = CalcRBF_Coeff(opt_sigma_x, subset_norm.T, vdx)
        rbf_coeff_y, _ = CalcRBF_Coeff(opt_sigma_y, subset_norm.T, vdy)
        output_dir_x.append(np.reshape(rbf_coeff_x, -1))
        output_dir_y.append(np.reshape(rbf_coeff_y, -1))
        
        opt_sigma_xs.append(opt_sigma_x)
        opt_sigma_ys.append(opt_sigma_y)
    
    if output_scalar != []:
        df_rbf_scalar = pd.DataFrame(np.transpose(output_scalar))
    else:
        df_rbf_scalar = pd.DataFrame()
        opt_sigmas = []
        
    if output_dir_x != []:
        df_rbf_dirx = pd.DataFrame(np.transpose(output_dir_x))
        df_rbf_diry = pd.DataFrame(np.transpose(output_dir_y))
    else:
        df_rbf_dirx = pd.DataFrame()
        df_rbf_diry = pd.DataFrame()
        opt_sigma_x, opt_sigma_y = [], []
        
    return df_rbf_scalar, df_rbf_dirx, df_rbf_diry, opt_sigmas, opt_sigma_x, opt_sigma_y

    
def RBF_Reconstruction(
    subset, ix_scalar_subset, ix_directional_subset,
    target, ix_scalar_target, ix_directional_target,
    dataset):
    '''
    Radial Basis Function (Gaussian) interpolator.

    subset                - subset used for fitting RBF (dim_input)
    ix_scalar_subset      - scalar columns indexes for subset
    ix_directional_subset - directional columns indexes for subset
    target                - target used for fitting RBF (dim_output)
    ix_scalar_target      - scalar columns indexes for target
    ix_directional_target - directional columns indexes for target
    dataset - dataset used for RBF interpolation (dim_input)
    '''

    
    
    subset_norm, mins, maxs = Normalize(
        subset, ix_scalar_subset, ix_directional_subset)
    
    # normalize subset and dataset
    dataset_norm, _, _ = Normalize(
        dataset, ix_scalar_subset, ix_directional_subset, mins, maxs)

    # output storage
    output = np.zeros((dataset.shape[0], target.shape[1] ))
    
    # RBF scalar variables 
    for ix in ix_scalar_target:
        
        # parameters
        d_sigma = 0
        sigma_min = 0.001
        sigma_max = 0.1
    
        v = target[:,ix]
        
        # minimize RBF cost function
        t0 = time.time()  # time counter
        
        # ensure that sigma opt is in the bounds of [sigma_min, sigma_max]
        while d_sigma < 0.0001:
            opt_sigma = fminbound(
                CostEps, sigma_min, sigma_max, args=(subset_norm.T, v)
            )
            lm_min = np.abs(opt_sigma-sigma_min)
            lm_max = np.abs(opt_sigma-sigma_max)
            
            if lm_min < 0.0001:
                sigma_min = sigma_min - sigma_min/2
                
            elif lm_max < 0.001:
                sigma_max = sigma_max + sigma_max/2
            
            d_sigma = np.nanmin([lm_min, lm_max])
            
        t1 = time.time()  # optimization time
        
        # calculate RBF coeff
        rbf_coeff, _ = CalcRBF_Coeff(opt_sigma, subset_norm.T, v)

        # RBF interpolation
        t2 = time.time()  # time counter
        output[:, ix] = RBF_Interpolation(
            opt_sigma, rbf_coeff, subset_norm.T, dataset_norm.T)
        t3 = time.time()  # interpolation time

        #print(
        #    'ix_scalar: {0},  optimization: {1:.2f} | interpolation: {2:.2f}'.format(
        #        ix, t1-t0, t3-t2)
        #)
        print('\rRange sigma {0:.4f}-{1:.4f} - Opt sigma {2:.4f}'.format(sigma_min, sigma_max, opt_sigma))

    # RBF directional variables
    for ix in ix_directional_target:
        
        # parameters
        d_sigma = 0
        sigma_min = 0.001
        sigma_max = 0.1
        
        v = target[:,ix]

        # x and y directional variable components
        vdg = np.pi/2 - v * np.pi/180
        pos = np.where(vdg < -np.pi)[0]
        vdg[pos] = vdg[pos] + 2 * np.pi
        vdx = np.cos(vdg)
        vdy = np.sin(vdg)

        # minimize RBF cost function
        t0 = time.time()  # time counter
        opt_sigma_x = fminbound(
            CostEps, sigma_min, sigma_max, args=(subset_norm.T, vdx)
        )
        opt_sigma_y = fminbound(
            CostEps, sigma_min, sigma_max, args=(subset_norm.T, vdy)
        )
        t1 = time.time()  # optimization time

        # calculate RBF coeff
        rbf_coeff_x, _ = CalcRBF_Coeff(opt_sigma_x, subset_norm.T, vdx)
        rbf_coeff_y, _ = CalcRBF_Coeff(opt_sigma_y, subset_norm.T, vdy)

        # RBF interpolation
        t2 = time.time()  # time counter
        output_x = RBF_Interpolation(
            opt_sigma_x, rbf_coeff_x, subset_norm.T, dataset_norm.T)
        output_y = RBF_Interpolation(
            opt_sigma_y, rbf_coeff_y, subset_norm.T, dataset_norm.T)
        t3 = time.time()  # interpolation time

        # join x and y components
        out = np.arctan2(output_y, output_x) * 180/np.pi
        out = 90 - out
        pos = np.where(out < 0)[0]
        out[pos] = out[pos] + 360
        output[:,ix] = out

        print(
            'ix_directional: {0},  optimization: {1:.2f} | interpolation: {2:.2f}'.format(
                ix, t1-t0, t3-t2)
        )
        print('opt sigma x {0}'.format(opt_sigma_x))
        print('opt_sigma_y {0}'.format(opt_sigma_y))

    return output

def RBF_Validation(
    subset, ix_scalar_subset, ix_directional_subset,
    target, ix_scalar_target, ix_directional_target,
    n_splits=3, shuffle=False):
    '''
    Radial Basis Function (Gaussian) k-fold mean squared error

    subset                - subset used for fitting RBF (dim_input)
    ix_scalar_subset      - scalar columns indexes for subset
    ix_directional_subset - directional columns indexes for subset
    target                - target used for fitting RBF (dim_output)
    ix_scalar_target      - scalar columns indexes for target
    ix_directional_target - directional columns indexes for target
    '''

    # get train-test combinations using kfold from sklearn
    kF = KFold(n_splits=n_splits, shuffle=shuffle, random_state=None)

    l_mse = []
    l_trn_ix = []
    l_tst_ix = []
    for c, (train_index, test_index) in enumerate(kF.split(subset)):
        print('RBFs Kfold Validation: {0}/{1}'.format(c+1, n_splits))

        # get train and test data
        X_train, X_test = subset[train_index], subset[test_index]
        Y_train, Y_test = target[train_index], target[test_index]

        # fit RBFs with train data and interpolate test data
        Y_rbf = RBF_Reconstruction(
            X_train, ix_scalar_subset, ix_directional_subset,
            Y_train, ix_scalar_target, ix_directional_target,
            X_test)

        # calculate mean squared error
        mse = mean_squared_error(Y_test, Y_rbf)
        print('mean squared error : {0}'.format(mse))
        print()

        # store data for output
        l_mse.append(mse)
        l_trn_ix.append(train_index)
        l_tst_ix.append(test_index)

    # return validation data
    return xr.Dataset(
        {
            'mean_squared_error': (('n_split',), l_mse),
            'train_index': (
                ('train','n_split',), np.column_stack(tuple(l_trn_ix))
            ),
            'test_index': (
                ('test','n_split',), np.column_stack(tuple(l_tst_ix))
            ),
        },
        coords = {
            'n_split':np.arange(n_splits),
        }
    )



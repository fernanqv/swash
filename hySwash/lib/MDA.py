#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import plotly.express as px

default_colors = px.colors.qualitative.Plotly

def Normalize(data, ix_scalar, ix_directional, minis=[], maxis=[]):
    '''
    Normalize data subset - norm = val - min) / (max - min)

    data - data to normalize, data variables at columns.
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    data_norm = np.zeros(data.shape) * np.nan

    # calculate maxs and mins 
    if minis==[] or maxis==[]:

        # scalar data
        for ix in ix_scalar:
            v = data[:, ix]
            mi = np.amin(v)
            ma = np.amax(v)
            data_norm[:, ix] = (v - mi) / (ma - mi)
            minis.append(mi)
            maxis.append(ma)

        minis = np.array(minis)
        maxis = np.array(maxis)

    # max and mins given
    else:

        # scalar data
        for c, ix in enumerate(ix_scalar):
            v = data[:, ix]
            mi = minis[c]
            ma = maxis[c]
            data_norm[:,ix] = (v - mi) / (ma - mi)

    # directional data
    for ix in ix_directional:
        v = data[:,ix]
        data_norm[:,ix] = v * np.pi 


    return data_norm, minis, maxis


def DeNormalize(data_norm, ix_scalar, ix_directional, minis, maxis):
    '''
    DeNormalize data subset for MaxDiss algorithm

    data - data to normalize, data variables at columns.
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    data = np.zeros(data_norm.shape) * np.nan

    # scalar data
    for c, ix in enumerate(ix_scalar):
        v = data_norm[:,ix]
        mi = minis[c]
        ma = maxis[c]
        data[:, ix] = v * (ma - mi) + mi

    # directional data
    for ix in ix_directional:
        v = data_norm[:,ix]
        data[:, ix] = v * 180 

    return data

def Normalized_Distance(M, D, ix_scalar, ix_directional):
    '''
    Normalized distance

    M -
    D -
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    dif = np.zeros(M.shape)

    # scalar
    for ix in ix_scalar:
        dif[:,ix] = D[:,ix] - M[:,ix]

    # directional
    for ix in ix_directional:
        ab = np.absolute(D[:,ix] - M[:,ix])
        dif[:,ix] = np.minimum(ab, 2*np.pi - ab)/np.pi

    dist = np.sum(dif**2,1)
    return dist

def MaxDiss_Simplified_NoThreshold(data, num_centers, ix_scalar, ix_directional):
    '''
    Normalize data and calculate centers using
    maxdiss simplified no-threshold algorithm

    data - data to apply maxdiss algorithm, data variables at columns
    num_centers - number of centers to calculate
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    # TODO: PARSED FROM MATLAB ORIGINAL. CAN REFACTOR

    print('\nMaxDiss waves parameters: {0} --> {1}\n'.format(
        data.shape[0], num_centers))

    # normalize scalar and directional data
    data_norm, minis, maxis = Normalize(data, ix_scalar, ix_directional, minis=[], maxis=[])
    
    # mda seed
    seed = np.where(data_norm[:,0] == np.amax(data_norm[:,0]))[0][0]

    # initialize centroids subset
    subset = np.array([data_norm[seed]])
    train = np.delete(data_norm, seed, axis=0)
 
    # repeat till we have desired num_centers
    n_c = 1
    while n_c < num_centers:
        m = np.ones((train.shape[0],1))
        m2 = subset.shape[0]

        if m2 == 1:
            xx2 = np.repeat(subset, train.shape[0], axis=0)
            d_last = Normalized_Distance(train, xx2, ix_scalar, ix_directional)

        else:
            xx = np.array([subset[-1,:]])
            xx2 = np.repeat(xx, train.shape[0], axis=0)
            d_prev = Normalized_Distance(train, xx2, ix_scalar, ix_directional)
            d_last = np.minimum(d_prev, d_last)

        qerr, bmu = np.amax(d_last), np.argmax(d_last)

        if not np.isnan(qerr):
            subset = np.append(subset, np.array([train[bmu,:]]), axis=0)
            train = np.delete(train, bmu, axis=0)
            d_last = np.delete(d_last, bmu, axis=0)

            # log
            fmt = '0{0}d'.format(len(str(num_centers)))
            print('   MDA centroids: {1:{0}}/{2:{0}}'.format(
                fmt, subset.shape[0], num_centers), end='\r')

        n_c = subset.shape[0]

    # normalize scalar and directional data
    print('\n')
    centroids = DeNormalize(subset, ix_scalar, ix_directional, minis, maxis)
    
    return(centroids)

def eliminate_nans(data):
    '''
    data: matrix vars in columns
    '''
    m, n = np.shape((data))
    for c in range(n):
        p_bool = np.isnan(data[:,c])
        data = data[p_bool==False, :]
        
    return data
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx    

def add_column(dataset, subset, num_centers, ix_scalar, ix_directional, id_column):
    'Add Hs values to df dataset'
    
    # To add hs to the centroids
    ixs_dataset = [i for i in (ix_scalar + ix_directional)]
    
    Hsp = []
    for j in range(len(subset)):
        lst1 = range(len(dataset[:,id_column]))
        for i, ix in enumerate(ixs_dataset):
            
            lst2 = np.where(np.round(dataset[:,ix],3) == np.round(subset[j, i],3))[0]
            lst1 = np.intersect1d(lst1, lst2)
        
        Hsp.append(lst1)
        # print(lst1)
    return(np.asarray(Hsp))

def proy_wind(orientation, xds_states):
    'Wind proyection over the shore-cross profile'
   
    rel_beta = np.nanmin([np.abs(xds_states.wdir-orientation),np.abs(xds_states.wdir+360-orientation)],axis=0)
    rad_beta = (rel_beta * np.pi)/180
    p_w = xds_states.w.values * np.cos(rad_beta)
    xds_states['wx'] = p_w
    
    return(xds_states)

# Calculate the point density
def density_scatter(x, y):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return(x, y, z)


def scatter_mda(data_samples, names, color=None, figsize=(15,15), ss=[10, 25]):
    
    # maximum data_samples = 2

    # scatterplot every variable (using matplotlib.gridspec)
    vnames = data_samples[0].columns
    fig, axs = plt.subplots(
        len(vnames)-1, len(vnames)-1, 
        figsize=figsize,
        sharex=False, sharey=False,
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for c1, v1 in enumerate(vnames[1:]):
        for c2, v2 in enumerate(vnames[:-1]):
            
            for pdata, data in enumerate(data_samples):
                if color == None:
                    axs[c2,c1].scatter(data[v1], data[v2], marker='o', s=ss[pdata], color=default_colors[pdata], alpha=0.6)
                else:
                    vmax = np.max([np.max(np.abs(c)) for c in color])
                    im = axs[c2,c1].scatter(data[v1], data[v2], marker='o', s=ss[pdata], c=color[pdata], alpha=0.6, cmap='bwr', vmin=-vmax, vmax=vmax)

            axs[c2,c1].set_facecolor('whitesmoke')
            axs[c2,c1].patch.set_alpha(0.7)
            axs[c2,c1].grid(c='w', linewidth=1.4)
            axs[c2,c1].set_axisbelow(True)
            if c1==c2:
                axs[c2,c1].set_xlabel(names[c1+1])
                axs[c2,c1].set_ylabel(names[c2])
            elif c1>c2:
                axs[c2,c1].xaxis.set_ticklabels([])
                axs[c2,c1].yaxis.set_ticklabels([])
                
            else:
                fig.delaxes(axs[c2, c1])
    if color != None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    
    return(fig)


def mda_polar(dataset, subset, cmap, figsize):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')      
    c = ax.scatter(np.deg2rad(dataset.values[:,1]), dataset.values[:,0], s = 5, c=dataset.values[:,0], cmap=cmap, alpha=0.75)
    ax.scatter(np.deg2rad(subset.values[:,1]), subset.values[:,0],  s=1, c='k', alpha=0.75)
    
    ax.set_title('$Representative$ $wind$ $states$', fontweight='bold')
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel('w (m/s)', size=13)  
    
    return(fig)

def one_scatter(dataset, subset, color, figsize):

    fig = plt.figure(figsize=figsize)
    plt.plot(dataset.hs.values, dataset.tp.values, '.', markersize=3, color=color)
    plt.plot(subset.hs.values, subset.tp.values, '.', markersize=5, color='k')
    
    plt.xlabel('$Hs$ $(m)$')
    plt.ylabel('$Tp$ $(s)$')
    plt.title('MaxDiss Classification of waves in Profile 17\n $Easthern$ $kwajalein$', fontweight='bold')
    
    return(fig)
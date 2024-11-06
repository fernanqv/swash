# Import python libraries
# basic
# import os

# 

# # common

# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from IPython.display import HTML
# mpl.rcParams['figure.dpi'] = 120

# addpath to bluemath modules
import os.path as op
import sys
import os
sys.path.insert(0, op.join(os.getcwd(),'hyswash/bluemath_tk'))


# wrap_swash modules
# from wrap_swash.wswash.wrap import SwashProject, SwashWrap, SwashInput
# from wrap_swash.wswash.postprocessor import Postprocessor
# from wrap_swash.wswash.plots import SwashPlot
# from wrap_swash.wswash.io import SwashIO
# from wrap_swash.wswash.profiles import reef

# statistical_toolkit  modules:
# from scipy.stats import qmc 
# from statistical_toolkit.bluemathtk.MDA import *
# from statistical_toolkit.bluemathtk.PCA import *
# from hyswash.lib.output_extract import *
# from hyswash.lib.waves import series_TMA
# from hyswash.lib.reconstruction import RBF_Reconstruction_singular, RBF_Reconstruction_spatial
#pip install plotly

# name of dimensions
name_dims = ['Hs', 'Hs_L0','plants']

# upper and lower bounds
low_bounds = [0.5,    0.005, 0] 
upp_bounds =[   3,     0.05, 3] 

# number of samples to obtain
n_dims = len(name_dims)
n_samples = 10000 

# JAVI: Pasar a función
# LHS execution. 
from scipy.stats import qmc 
sampler = qmc.LatinHypercube(d = n_dims, seed=1)
dataset = sampler.random(n = n_samples)
dataset = qmc.scale(dataset, low_bounds, upp_bounds)

# convert to dataframe
import pandas as pd
df_dataset = pd.DataFrame(data=dataset, columns=name_dims)

#dims = df_dataset.columns.tolist()

# JAVI: Donde está el scatter que estaba en data
#import bluemath_tk.core.data
#fig = bluemath_tk.core.data.scatter(df_dataset)

# MDA
# JAVI: La carpeta datamining no tiene __init__.py
import bluemath_tk.datamining.mda
mda_hy = bluemath_tk.datamining.mda.MDA(data=df_dataset)
mda_hy.run(10)
mda_hy.scatter_data(plot_centroids=True)


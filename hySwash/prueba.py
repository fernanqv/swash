
import os.path as op
import sys
import os
sys.path.insert(0, op.join(os.getcwd(),'swash/hySwash/'))

# VALVA. VARIABLES QUE VAN A SER PARAMETRIZADAS
# name of dimensions
name_dims = ['Hs', 'Hs_L0','plants']

# upper and lower bounds
low_bounds = [0.5,    0.005, 0] 
upp_bounds =[   3,     0.05, 3] 

# number of samples to obtain
n_dims = len(name_dims)
n_samples = 10000 


# LHS execution. # VALVA. PASAR A BLUEMATH_TK
from scipy.stats import qmc 
sampler = qmc.LatinHypercube(d = n_dims, seed=1)
dataset = sampler.random(n = n_samples)
dataset = qmc.scale(dataset, low_bounds, upp_bounds)

# convert to dataframe
import pandas as pd
df_dataset = pd.DataFrame(data=dataset, columns=name_dims)

#dims = df_dataset.columns.tolist()


import bluemath_tk.core.data
fig = bluemath_tk.core.data.scatter(df_dataset)

# MDA # VALVA. PASAR A BLUEMATH_TK

import bluemath_tk.datamining.mda
mda_hy = bluemath_tk.datamining.mda.MDA(data=df_dataset)
mda_hy.run(10)
df_subset=mda_hy.centroids
#mda_hy.scatter_data(plot_centroids=True)


import wrap_swash.wswash.wrap
import wrap_swash.wswash.io
import wrap_swash.wswash.plots
import wrap_swash.wswash.profiles

# from wrap_swash.wswash.wrap import SwashProject, SwashWrap, SwashInput
# from wrap_swash.wswash.postprocessor import Postprocessor
# from wrap_swash.wswash.plots import SwashPlot
# from wrap_swash.wswash.io import SwashIO
# from wrap_swash.wswash.profiles import reef
# Initialize the project
p_proj = op.abspath(op.join(os.getcwd(), 'projects_valva')) # swash projects main directory
n_proj = 'test_valva'                                             # project name

sp = wrap_swash.wswash.wrap.SwashProject(p_proj, n_proj)
sw = wrap_swash.wswash.wrap.SwashWrap(sp)
si = wrap_swash.wswash.io.SwashIO(sp)
sm = wrap_swash.wswash.plots.SwashPlot(sp)

# Set the simulation period and grid resolution
sp.tendc = 1800                          # simulation period (SEC)
sp.warmup = 0.15 * sp.tendc              # spin-up time (s) (default 15%)
sp.b_grid.dx = 1            # bathymetry mesh resolution at x axes (m)
sp.dxL = 40                 # nº nodes per wavelength
sp.dxinp = 1                # bathymetry spacing resolution (m
sp.dyinp = 1

h0 = 15                     # offshore depth (m)
Slope1 = 0.05               # fore shore slope
Slope2 = 0.1                # inner shore slope
Wreef = 200                 # reef bed width (m)
Wfore = 500                 # flume length before fore toe (m)
bCrest = 10                 # beach heigh (m)
emsl = 2.5                  # mean sea level (m)

depth = wrap_swash.wswash.profiles.reef(sp.b_grid.dx, h0, Slope1, Slope2, Wreef, Wfore, bCrest, emsl)
sp.set_depth(depth, sp.dxinp, sp.dyinp)
fig = sm.plot_depthfile()
fig.show()

sp.friction_file = False
sp.friction = True
sp.Cf = 0.01                       # manning frictional coefficient  (m^-1/3 s)
sp.cf_ini = 0                      # first point along the profile 
sp.cf_fin = 800                    # last point along the profile 

# Create wave series and save 'waves.bnd' file
sp.deltat = 1              # delta time over which the wave series is defined

sp.non_hydrostatic = True  # True or False
sp.vert = 1                # vertical layers
sp.delttbl = 1             # time between output fields (s)df_subset['H'] = df_subset['Hs']

import numpy as np
df_subset['H'] = df_subset['Hs']
df_subset['T'] = np.sqrt((df_subset['Hs'].values * 2 * np.pi) / (9.806 * df_subset['Hs_L0']))
df_subset['gamma'] = 2
df_subset['warmup'] = sp.warmup
df_subset['deltat'] = sp.deltat
df_subset['tendc'] = sp.tendc
df_subset['WL'] = 1

# Create list of swash wrappers

import hyswash.lib.waves

list_wrap = []
for iw, waves in df_subset.iterrows():
    series = hyswash.lib.waves.series_TMA(waves, depth[0])
    si = wrap_swash.wswash.wrap.SwashInput() 
    si.waves_parameters = waves
    si.waves_series = series
    list_wrap.append(si)

sw = wrap_swash.wswash.wrap.SwashWrap(sp)
waves = sw.build_cases(list_wrap)






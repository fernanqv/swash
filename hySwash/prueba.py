
import os.path as op
import sys
import os
sys.path.insert(0, op.join(os.getcwd(),'swash/hySwash/'))

# VALVA. VARIABLES QUE VAN A SER PARAMETRIZADAS
# name of dimensions
name_dims = ['Hs', 'Hs_L0','VegetationHeight']

# upper and lower bounds
low_bounds = [0.5,    0.005, 0] 
upp_bounds =[   3,     0.05, 1.5] 

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
#fig = bluemath_tk.core.data.scatter(df_dataset)

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
p_proj = op.abspath(op.join(os.getcwd(), 'projects')) # swash projects main directory
n_proj = 'test_valva'                                             # project name

sp = wrap_swash.wswash.wrap.SwashProject(p_proj, n_proj)
si = wrap_swash.wswash.wrap.SwashInput() 
sw = wrap_swash.wswash.wrap.SwashWrap(sp)
sm = wrap_swash.wswash.plots.SwashPlot(sp)

###############################
# 10 CONFIGURACION DE PARAMETROS
# Set the simulation period and grid resolution
sp.tendc = 1800                          # simulation period (SEC)
sp.Cf = 0.01                       # manning frictional coefficient  (m^-1/3 s)
sp.vert = 3                # vertical layers


#sp.height = 1.148260794155014                       # plant height per vertical segment (m)


sp.warmup = 0.15 * sp.tendc              # spin-up time (s) (default 15%)
sp.deltat = 1              # delta time over which the wave series is defined
sp.b_grid.dx = 1            # bathymetry mesh resolution at x axes (m)
sp.dxL = 20                 # nº nodes per wavelength
sp.dxinp = 1                # bathymetry spacing resolution (m)
sp.dyinp = 1

# FRICCION
sp.friction_file = False
sp.friction = True
sp.cf_ini = 0                      # first point along the profile 
sp.cf_fin = 800                    # last point along the profile 

# Create wave series and save 'waves.bnd' file
sp.non_hydrostatic = True  # True or False

sp.delttbl = 1             # time between output fields (s)df_subset['H'] = df_subset['Hs']

# PASAMOS DEL VIENTO. SE PUEDE?
# Define wind parameters
sp.wind = False           # wind direction at 10 m height (º)

# vegetation
# VEGEtation < [height] [diamtr] [nstems] [drag] > INERtia [cm] POROsity Vertical
# VEGETATION 1.148260794155014 0.5009019570650376 1 1.0
sp.vegetation = 1                      # bool: activate vegetation
sp.diamtr = 0.5009019570650376                       # plant diameter per vertical segment (m)
sp.nstems = 1                       # num of plants per square meter for each segment
sp.drag = 1.0                         # drag coefficient per vertical segment
sp.vegetation_file = True            # bool: use a vegetation file
sp.np_ini = 800                     # vegetation start cell
sp.np_fin = 1000                       # vegetation end cell

###############################

# 11 DEFINIR LA BATIMETRÍA --> fichero depth.bot (estático)
h0 = 15                     # offshore depth (m)
Slope1 = 0.05               # fore shore slope
Slope2 = 0.1                # inner shore slope
Wreef = 200                 # reef bed width (m)
Wfore = 500                 # flume length before fore toe (m)
bCrest = 10                 # beach heigh (m)
emsl = 2.5                  # mean sea level (m)

# 
depth = wrap_swash.wswash.profiles.reef(sp.b_grid.dx, h0, Slope1, Slope2, Wreef, Wfore, bCrest, emsl)
sp.set_depth(depth, sp.dxinp, sp.dyinp)
fig = sm.plot_depthfile()
fig.show()





import numpy as np
df_subset['H'] = df_subset['Hs']
df_subset['T'] = np.sqrt((df_subset['Hs'].values * 2 * np.pi) / (9.806 * df_subset['Hs_L0']))
df_subset['gamma'] = 2
df_subset['warmup'] = sp.warmup
df_subset['deltat'] = sp.deltat
df_subset['tendc'] = sp.tendc
df_subset['WL'] = 1
df_subset['height'] = df_subset['VegetationHeight']

# Create list of swash wrappers

import lib.waves

# TAREA. Pasar a una función para crear ficheros de entrada
list_wrap = []
i=0
for iw, waves in df_subset.iterrows():
    series = lib.waves.series_TMA(waves, depth[0]) # PABLO: ¿qué significa el depth[0]?
    si.waves_parameters = waves
    si.waves_series = series
    si.height =  df_subset['height'][i]
    list_wrap.append(si)
    i = i + 1
    # plant height per vertical segment [m]     # si.nstems = df_subset['Dcoral'].values[i]      # nº plants stands per square meter
    # si.ncoral = df_subset['Ncoral'].values[i]
    # si.np_ini = 800               # first point along the profile 
    # si.np_fin = si.np_ini + Wreef             # last point along the profile 
    # si.w_steps = df_subset['Wcoral'].values[i]
    # list_wrap.append(si)



# TAREA. Lanzar trabajos
sw = wrap_swash.wswash.wrap.SwashWrap(sp)
waves = sw.build_cases(list_wrap)
sw.run_cases()






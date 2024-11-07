#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# swash wrap module
from wswash.wrap import SwashProject, SwashWrap, SwashInput
from wswash.waves import series_Jonswap


# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))


# SWASH project
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = 'demo_04'                    # project name

# SWASH wrap objects
sp = SwashProject(p_proj, n_proj)
sw = SwashWrap(sp)


# set bathymetry
depth = np.linspace(11, -2, 601)
dxinp = 1  # bathymetry x spacing resolution (m)
dyinp = 1  # bathymetry y spacing resolution (m)

sp.set_depth(depth, dxinp, dyinp)


# SWASH project configuration
sp.non_hydrostatic = True  # True or False
sp.vert = 1                # vertical layers
sp.delttbl = 0.1           # output time interval between fields
sp.dxL = 30                # nº nodes per wavelength


# SWASH case
waves_parameters = {
    "H": 1,                # wave height (m) 
    "WL": 0,               # water level (m)
    "T": 10,               # wave period of frequency component 1
    "gamma": 3,            # jonswap peak parameter
    "warmup": 800,
    "tendc": 4400,
    "deltat": 0.5,
}

# make waves series dataset
waves_series = series_Jonswap(waves_parameters)


# SWASH case input 
si = SwashInput()
si.waves_parameters = waves_parameters
si.waves_series = waves_series


# build cases
sw.build_cases([si])

# run cases
sw.run_cases()

# postprocess case output
output = sw.postprocessing(case_ix=0)

print('output')
for k in output.keys():
    print(k)
    print(output[k])
    print()


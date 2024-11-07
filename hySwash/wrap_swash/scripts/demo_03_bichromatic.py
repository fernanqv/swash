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
from wswash.waves import series_regular_bichromatic


# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))


# SWASH project
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = 'demo_03'                    # project name

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
    "H": 0.5,              # wave height (m) 
    "WL": 3,               # water level (m)
    "T1": 4,               # wave period of frequency component 1
    "T2": 4.5,             # wave period of frequency component 2
    #"warmup": 200,
    #"tendc": 1000,
    "warmup": 10,
    "tendc": 40,
    "deltat": 0.5,
}

# make waves series dataset
waves_series = series_regular_bichromatic(waves_parameters)

# Define wind parameters
Vel = 14        # wind speed  at 10 m height (m/s)
Wdir = 0        # wind direction at 10 m height (º)
Ca = 0.0026     # dimensionless coefficient (default 0.002)

wind = {
    "wdir": Wdir,
    "vx": Vel,
    "Ca": Ca,
}


# SWASH case input 
si = SwashInput()
si.waves_parameters = waves_parameters
si.waves_series = waves_series
si.wind = wind


# build cases
sw.build_cases([si])

# run cases
sw.run_cases()

# postprocess case output
output = sw.postprocessing(case_ix=0, do_spectral_analysis=True)

print('\noutput\n')
for k in output.keys():
    print(k)
    print(output[k])
    print()


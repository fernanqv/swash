#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import xarray as xr

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# swash wrap module
from wswash.wrap import SwashProject, SwashWrap, SwashInput
from wswash.waves import series_Jonswap


# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

# test data: csiro point 
p_waves_demo = op.join(p_demo, 'waves_csiro_demo.nc')


# SWASH project
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = 'demo_01'                    # project name

# SWASH wrap objects
sp = SwashProject(p_proj, n_proj)
sw = SwashWrap(sp)


# set bathymetry
depth = np.ones((31,)) * 0.4
dxinp = 1  # bathymetry x spacing resolution (m)
dyinp = 1  # bathymetry y spacing resolution (m)

sp.set_depth(depth, dxinp, dyinp)


# SWASH project configuration
sp.non_hydrostatic = True  # True or False
sp.vert = 1                # vertical layers
sp.delttbl = 0.1           # output time interval between fields
sp.dxL = 30                # nº nodes per wavelength


# SWASH cases: multiple waves sea states
waves = xr.open_dataset(p_waves_demo)
waves = waves.squeeze()   # remove lon,lat dim (len=1)

waves = waves.resample(time='1D').pad()
waves = waves.isel(time = [1, 50, 100, 150])

WL = 10       # water level
tendc = 30    # simualtion period (SEC)
warmup = 5    # spin-up time (s) 
deltat = 0.5  # wave series delta time
gamma = 1.9   # jonswap spectra gamma


# generate one case for each wave sea state
l_cases = []

for i, t in enumerate(waves.time):
    w = waves.sel(time=t)

    waves_parameters = {
        "H": float(w.hs.values),
        "T": float(w.t.values),
        "WL": WL,
        "gamma": gamma,
        "warmup": warmup,
        "tendc": tendc,
        "deltat": deltat,
    }

    # make waves series dataset
    waves_series = series_Jonswap(waves_parameters)

    # SWASH case input 
    si = SwashInput()
    si.waves_parameters = waves_parameters
    si.waves_series = waves_series

    l_cases.append(si)


# build cases
sw.build_cases(l_cases)

# run cases
sw.run_cases()

# postprocess case output
output = sw.postprocessing(case_ix=0, do_spectral_analysis=True)

print('\noutput\n')
for k in output.keys():
    print(k)
    print(output[k])
    print()


#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO
# - codigo obsoleto
# - necesito metodologia build_cases_spec para acoplar desde SWASH


# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# SWASH wrap
from wswash.wrap import SwashProject, SwashWrap

'''
PURPOSE OF TEST: test of coupling output SWAN - SWASH
C1a: Hm0 = 0.3 m, Tp = 8 s, dspr = 2 degr
horizontal bottom (depth = 5 m)
'''

# --------------------------------------------------------------------------- #
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

# --------------------------------------------------------------------------- #
# SWASH project (config bathymetry, parameters, computational grid)
p_proj = op.join(p_data, 'projects')      # swash projects main directory
n_proj = '01_coupling'                    # project name

sp = SwashProject(p_proj, n_proj)
sp.Nonhydrostatic = True
sp.wind = False
sp.specfilename = 't11cou01.sp2'

# parameters (sea level, jonswap gamma)
sp.par_jonswap_gamma = 1.9
sp.par_sea_level = 5
sp.vert = 1           # vertical layers
sp.tbegc = 0          # initial time (compute)
sp.deltc = 0.1        # time step (compute)
sp.tendc = 2000       # final time (compute)
sp.tbegtbl = 500      # initial time in fields output
sp.delttbl = 0.1      # time interval between fields
sp.delttbl_ = 'SEC'   # time units
sp.nlocs = 5          # number stations along curve (-1)

# configure SWASH bathymetry
sp.depth = np.ones((2, 2)) * 5
sp.dp_xpc = 5000      # x origin
sp.dp_ypc = 5000      # y origin
sp.dp_alpc = 0        # x-axis direction 
sp.dp_xlenc = 1       # grid length in x
sp.dp_ylenc = 0       # grid length in y
sp.dp_mxc = 1         # number mesh x
sp.dp_myc = 1         # number mesh y
sp.dp_dxinp = 100     # size mesh x
sp.dp_dyinp = 250     # size mesh y

# SWASH Computational grid
sp.cc_xpc = 5000      # x origin
sp.cc_ypc = 5000      # y origin
sp.cc_alpc = 0.00     # x-axis direction 
sp.cc_xlenc = 100     # grid length in x
sp.cc_ylenc = 250     # grid length in y
sp.cc_mxc = 50        # number mesh x
sp.cc_myc = 125       # number mesh y
sp.cc_dxinp = 10000   # size mesh x
sp.cc_dyinp = 0       # size mesh y

# output request
sp.x_out = 5050
sp.y_out = 5050

# --------------------------------------------------------------------------- #
# SWASH case input: waves_event 

# Wave event
p_spec = op.join(p_demo, sp.specfilename) # swan spectrum path
# --------------------------------------------------------------------------- #
# SWASH wrap (create case files, launch SWASH num. model, extract output)
sw = SwashWrap(sp)

# build cases from waves data
sw.build_cases_spec(p_spec)

# run SWASH
sw.run_cases()

# extract output from cases (curved output)
xds_table = sw.extract_output_points()
print('\n')
print(xds_table)

# Plot spectrum in output point
f, E = sw.make_spectrum()
print('\noutput')


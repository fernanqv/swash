#!/usr/bin/env python
# -*- coding: utf-8 -*-


# common
import os
import os.path as op

# pip
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# swash module
from .plots import SwashPlot

def parse_time(s):
    'Convert seconds time to hours, minutes and seconds'

    sec = timedelta(seconds = s)
    d = datetime(1,1,1) + sec
    h = d.hour + (d.day-1)*24

    return (h, d.minute, d.second)

def read_tabfile(p_file):
    'Read .tab file and return pandas.DataFrame'

    # read head colums (variables names)
    f = open(p_file, "r")
    lines = f.readlines()

    names = lines[4].split()
    names = names[1:] # Eliminate '%'

    # read data rows
    values = pd.Series(lines[7:]).str.split(expand=True).values.astype(float)
    df = pd.DataFrame(values, columns=names)

    f.close()

    return(df)


class SwashIO(object):
    'SWASH numerical model input/output'

    def __init__(self, swash_proj):
        self.proj = swash_proj
        self.plots = SwashPlot(self.proj)   # swash plotting tool

    def make_project(self):
        'Make swash project folder and subfolders'

        if not op.isdir(self.proj.p_cases): os.makedirs(self.proj.p_cases)

    def build_case(self, case_id, swash_input):
        '''
        Build SWASH case input files

        case_id  - SWASH case index (int)
        swash_input - SWASH input object
        '''
        print("hola")
        # SWASH case path
        p_case = op.join(self.proj.p_cases, case_id)

        # make execution dir
        if not op.isdir(p_case): os.makedirs(p_case)

        # make depth file
        self.make_depth(op.join(p_case, 'depth.bot'))

        # make wave boundary file
        self.make_waves_series(op.join(p_case, 'waves.bnd'), swash_input.waves_series)

        # make friction file
        if self.proj.friction_file == True:
            self.make_friction(op.join(p_case, 'friction.txt'))

        # make vegetation  file
        if self.proj.vegetation_file == True:
            self.make_vegetation(op.join(p_case, 'plants.txt'))

        # make input.swn file
        self.make_input(op.join(p_case, 'input.sws'), case_id, swash_input)

        # store a copy of swash_input at case folder
        swash_input.save(op.join(p_case, 'swash_input.pkl'))

    def make_depth(self, p_file):
        'Export depth_grid to plain text depth file (SWASH compatible)'

        np.savetxt(p_file, self.proj.depth, fmt='%.2f')

    def make_waves_series(self, p_file, waves_series):
        '''
        Generates waves series boundary file

        waves_series - numpy.array 2D with time and elevation
        '''

        np.savetxt(p_file, waves_series, fmt='%f %e')

    def make_friction(self, p_file):
        'writes friction file'

        depth = self.proj.depth
        cf_ini = self.proj.cf_ini
        cf_fin = self.proj.cf_fin
        cf = self.proj.cf

        mx = len(depth)

        f1 = np.zeros((int(cf_ini))) + 0
        f2 = np.zeros((int(cf_fin - cf_ini))) + cf
        f3 = np.zeros((int(mx - cf_fin))) + 0

        val = np.concatenate([f1, f2, f3])[np.newaxis].T

        # Export and save
        np.savetxt(p_file, val, fmt='%f')

    def make_vegetation(self, p_file):
        'writes vegetation file'

        depth = self.proj.depth
        np_ini = self.proj.np_ini
        np_fin = self.proj.np_fin
        nstems = self.proj.nstems

        mx = len(depth)

        f = pd.DataFrame({})
        f1 = np.zeros((int(np_ini))) + 0
        f2 = np.zeros((int(np_fin - np_ini))) + nstems
        f3 = np.zeros((int(mx - np_fin))) + 0

        val = np.concatenate([f1, f2, f3])[np.newaxis].T

        # Export and save
        np.savetxt(p_file, val, fmt='%e') 

    def make_input(self, p_file, id_run, swash_input):
        '''
        Writes input.sws file from waves sea state for computation

        p_file            - input.sws file path
        waves_parameters  - wave parameters (hs, tp, ...)
        bnd               - wave sea state active boundaries

        more info: http://swash.sourceforge.net/download/zip/swashuse.pdf
        '''

        # get data from swash_input
        waves_params = swash_input.waves_parameters
        wind = swash_input.wind

        # waves sea state parameters
        WL = waves_params['WL']          # water level
        warmup = waves_params['warmup']  # spinup time 
        tendc = waves_params['tendc']    # simulation time 

        # .sws file parameters
        vert = self.proj.vert
        non_hydrostatic = self.proj.non_hydrostatic 

        # friction (entire bottom or file)
        friction_bottom = self.proj.friction_bottom
        cf = self.proj.cf
        friction_file = self.proj.friction_file

        # vegetation
        vegetation = self.proj.vegetation
        vegetation_file = self.proj.vegetation_file
        height = swash_input.height
        #height = self.proj.height
        diamtr = self.proj.diamtr
        nstems = self.proj.nstems
        drag = self.proj.drag

        # output
        tbegtbl = self.proj.tbegtbl
        delttbl = self.proj.delttbl
        delttbl_ = self.proj.delttbl_

        # bathymetry depth values
        depth = self.proj.depth

        # bathymetry grid  (from swash project)
        bg = self.proj.b_grid

        # computational grid (from swash input)
        cg = swash_input.c_grid

        # computational time step
        c_timestep = swash_input.c_timestep

        # computation duration (parsed to hours, minutes, seconds)
        duration = tendc + warmup
        Hr, Mn, Sc = parse_time(duration)

        # .sws text file
        t = "$Project name\n"
        t += "PROJECT '{0}' '{1}'\n$\n".format(self.proj.name, id_run)

        t += "$Set water level\n"
        t += 'SET LEVEL={0}\n$\n'.format(WL)

        # MODE: requests a 1D-mode / 2D-mode of SWASH
        t += "$(1D-mode, flume) or (2D-mode, basin)\n"
        t += 'MODE DYNanic ONED\n'

        # to choose between Cartesian and spherical coordinates
        t += 'COORD CARTesian\n'

        # computational grid
        t += "$Computational grid: geographic location, size, resolution and orientation\n"
        t += 'CGRID {0} {1} {2} {3} {4} {5} {6}\n$\n'.format(
            cg.xpc, cg.ypc, cg.alpc, cg.xlenc, cg.ylenc, cg.mxc, cg.myc)

        t += "$Multi-layered mode\n"
        t += 'VERT {0}\n$\n'.format(vert)

        # bathymetry
        t += "$Reading bathymetry values from file\n"
        t += 'INPGRID BOTTOM {0} {1} {2} {3} {4} {5:.2f} {6}\n'.format(
            bg.xpc, bg.ypc, bg.alpc, bg.mxc, bg.myc, bg.dx, bg.dy)
        t += "READINP BOTTOM 1 '{0}' 1 0 FREE\n$\n".format('depth.bot')

        # vegetation file
        if vegetation_file:
            t += "$Reading vegetation values from file\n"
            t += 'INPGRID NPLANTS {0} {1} {2} {3} {4} {5} {6}\n'.format(
                bg.xpc, bg.ypc, bg.alpc, bg.mxc, bg.myc, bg.dx, bg.dy)
            t += "READINP NPLANTS 1 '{0}' 1 0 FREE\n$\n".format('plants.txt')

        # friction file
        if friction_file:
            t += "$Reading friction values from file\n"
            t += 'INPGRID FRICTION {0} {1} {2} {3} {4} {5} {6}\n'.format(
                bg.xpc, bg.ypc, bg.alpc, bg.mxc, bg.myc, bg.dx, bg.dy)
            t += "READINP FRICTION 1 '{0}' 1 0 FREE\n$\n".format('friction.txt')
            t += "FRIC MANNING\n"

        t += "$Initial values for flow variables\n"
        t += 'INIT ZERO\n'
        t += "$\n"

        # waves 
        t += "$Hydraulic boundary conditions\n"
        t += "BOU SIDE W CCW BTYPE WEAK CON SERIES 'waves.bnd'\n"
        t += 'BOU SIDE E CCW BTYPE RADIATION \n'
        t += 'SPON E 10 \n'

        t += "$\n"

        # numerics
        t += "$Physics\n"
        t += 'BREAK\n'

        # friction entire bottom
        if friction_bottom:
            t += "FRIC MANNING {0}\n".format(cf)

        # vegetation parameters
        # TODO repasar funcionamiento "nstems" en combinacion con vegetation_file
        if vegetation:
            t += "VEGETATION {0} {1} {2} {3}\n".format(height, diamtr, nstems, drag)

        # wind
        if wind != None:

            # default drag coefficient
            if not 'Ca' in wind.keys(): wind['Ca'] = 0.002

            t += "WIND {0} {1} CONSTANT {2}\n$\n".format(
                wind['vx'], wind['wdir'], wind['Ca'],
            )

        if non_hydrostatic:
            t += "$Numerics\n"
            t += 'NONHYDrostatic\n$\n'

        if not wind:
            t += "$Output quantities\n"
            t += 'DISCRET UPW MOM\n'
            t += 'DISCRET UPW UMOM H NONE\n'
            t += 'DISCRET UPW WMOM H NONE\n'
            t += 'DISCRET CORR\n$\n'

        t += "$Time integration\n"
        t += 'TIMEI 0.1 0.5\n$\n'

        t += 'QUANTITY XP hexp=10\n'
        t += 'QUANT RUNUP delrp 0.01\n$\n'

        # output
        t += "CURVE 'line' {0} {1} {2} {3} {4}\n".format(
            cg.xpc, cg.ypc, cg.xlenc, cg.xpc + cg.xlenc, cg.ypc + cg.ylenc)

        # general output
        t += "TABLE 'line' HEAD 'output.tab' TSEC XP YP BOTL WATL QMAG OUTPUT {0} {1} {2}\n$\n".format(
            tbegtbl, delttbl, delttbl_)

        # runup
        t += "TABLE 'NOGRID' HEAD 'run.tab' TSEC RUNUP OUTPUT {0} {1} {2}\n$\n".format(
            tbegtbl, delttbl, delttbl_)

        # compute
        t += "$Starts computation\n"
        t += 'TEST  1,0\n'
        t += 'COMPUTE 000000.000 {0:.4f} SEC {1}{2}{3}.000\n'.format(
            c_timestep, str(Hr).zfill(2), str(Mn).zfill(2), str(Sc).zfill(2))
        t += 'STOP\n$\n'

        # write file:
        with open(p_file, 'w') as f:
            f.write(t)

    def print_input(self, case_ix=0):
        '''
        Print "input.sws" text

        case_ix - case index (int)
        '''

        p_cases = self.proj.p_cases
        case_id = '{0:04d}'.format(case_ix)

        path = op.join(p_cases, case_id)
        os.chdir(path)
        f = open('input.sws')

        lines = f.read().splitlines()
        f.close()
        for line in lines:
            print(line)

    def output_points(self, p_case, case_id):
        'read run.tab and output.tab files and returns xarray.Dataset'

        # read output.tab and run.tab to pandas.DataFrame
        p_dat = op.join(p_case, 'output.tab')
        p_run = op.join(p_case, 'run.tab')

        ds1 = read_tabfile(p_dat)
        ds2 = read_tabfile(p_run)

        # parse pandas.DataFrame to xarray.Dataset
        ds1 = ds1.set_index(['Xp', 'Yp','Tsec']) #, coords = Time, Xp, Yp
        ds1 = ds1.to_xarray()

        ds2 = ds2.set_index(['Tsec']) #, coords = Time, Xp, Yp
        ds2 = ds2.to_xarray()

        # merge output files to one xarray.Dataset
        ds = xr.merge([ds1, ds2], compat='no_conflicts')

        # assign correct coordinate case_id
        ds.coords['case_id'] = case_id
        
        return(ds)


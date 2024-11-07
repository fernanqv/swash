#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import sys
import os
import os.path as op
import subprocess as sp

# pip
import numpy as np
import pickle

# swash
from .io import SwashIO
from .plots import SwashPlot
from .waves import waves_dispersion


class SwashGrid(object):
    'SWASH numerical model grid (bathymetry and computation)'

    def __init__(self):

        self.xpc = None      # x origin
        self.ypc = None      # y origin
        self.alpc = None     # x-axis orientation
        self.xlenc = None    # grid length in x
        self.ylenc = None    # grid length in y
        self.mxc = None      # number mesh x
        self.myc = None      # number mesh y
        self.dx = None       # size mesh x
        self.dy = None       # size mesh y


class SwashInput(object):
    'SWASH numerical model case input'

    def __init__(self):

        # waves boundary conditions
        self.waves_parameters = None  # waves parameters  (dict: T, H, ...)
        self.waves_series = None      # waves series input (numpy array: time, elevation)

        # wind conditions
        self.wind = None              # winds input (dict: wdir, vx, Ca)

        # computational time step and grid
        self.c_timestep = None        # computational time step
        self.c_grid = SwashGrid()     # computational grid

    def prepare_computational_grid(self, nodes_per_wavel, depth0, b_grid):
        '''
        Prepares computational grid and computational time step

        nodes_per_wavel  - nodes per wavelength
        depth0           - depth at waves generation (m)
        b_grid           - swash project bathymetry grid
        '''

        # get minor wave period if bichromatic waves series
        if 'T' not in self.waves_parameters.keys():
            T = np.min([self.waves_parameters['T1'], self.waves_parameters['T2']])

        else:
            T = self.waves_parameters['T']  # waves period

        # Assuming there is always 1m of setup due to (IG, VLF)
        Ls, ks, cs = waves_dispersion(T, 1)  # 1 meter depth for infragravity waves 
        L, k, c = waves_dispersion(T, depth0)

        # calculate computational time step and grid spacing
        comp_dx = Ls / nodes_per_wavel
        comp_step = 0.5 * comp_dx / (np.sqrt(9.806*depth0) + np.abs(c))

        self.c_timestep = comp_step

        # prepare computational grid parameters (same as bathymetry grid)
        self.c_grid.xpc = b_grid.xpc
        self.c_grid.ypc = b_grid.ypc
        self.c_grid.alpc = b_grid.alpc
        self.c_grid.xlenc = b_grid.xlenc
        self.c_grid.ylenc = b_grid.ylenc
        self.c_grid.dy = b_grid.dy
        self.c_grid.myc = b_grid.myc

        # computational grid modifications
        self.c_grid.dx = comp_dx
        self.c_grid.mxc = int(b_grid.mxc / comp_dx)

    def save(self, p_save):
        'Stores a copy of this input to p_save'

        with open(p_save, 'wb') as f:
            pickle.dump(self, f)


class SwashProject(object):
    'SWASH numerical model project parameters. used inside swashwrap and swashio'

    def __init__(self, p_proj, n_proj):
        '''
        SWASH project information will be stored here

        http://swash.sourceforge.net/download/zip/swashuse.pdf
        '''

        # project name and paths
        self.name = n_proj                       # project name
        self.p_main = op.join(p_proj, n_proj)    # project path
        self.p_cases = op.join(self.p_main)      # project cases

        # input friction
        self.friction_bottom = None              # bool: activate friction (entire  bottom)
        self.cf = None                           # friction manning coefficient (m^-1/3 s)
        self.friction_file = None                # bool: use a friction file
        self.cf_ini = None                       # friction start cell
        self.cf_fin = None                       # friction end cell

        # vegetation
        self.vegetation = None                   # bool: activate vegetation
        self.height = None                       # plant height per vertical segment (m)
        self.diamtr = None                       # plant diameter per vertical segment (m)
        self.nstems = None                       # num of plants per square meter for each segment
        self.drag = None                         # drag coefficient per vertical segment
        self.vegetation_file = None              # bool: use a vegetation file
        self.np_ini = None                       # vegetation start cell
        self.np_fin = None                       # vegetation end cell

        # computational grid parameters
        self.dxL = None                          # nº of nodes per wavelength

        # bathymetry grid and depth
        self.b_grid = SwashGrid()                # bathymetry grid
        self.depth = None                        # bathymetry deph values (1D numpy.array)

        # input.swn file parameters
        self.vert = None                         # multilayered mode 
        self.non_hydrostatic = False             # non hydrostatic pressure

        # output default configuration
        self.tbegtbl = 0                         # initial time in fields output
        self.delttbl = 1                         # time interval between fields
        self.delttbl_ = 'SEC'                    # time units

    def set_depth(self, depth, dx, dy):
        '''
        set depth values and generates bathymetry grid parameters

        depth - bathymetry depth value (2D numpy.array)
        dx - bathymetry x spacing resolution
        dy - bathymetry y spacing resolution
        '''

        # set depth values
        self.depth = depth

        # set bathymetry grid parameters
        self.b_grid.xpc = 0
        self.b_grid.ypc = 0
        self.b_grid.alpc = 0
        self.b_grid.dx = dx
        self.b_grid.dy = dy
        self.b_grid.mxc = len(depth) - 1
        self.b_grid.myc = 0
        self.b_grid.xlenc = int(self.b_grid.mxc * dx)
        self.b_grid.ylenc = 0


class SwashWrap(object):
    'SWASH numerical model wrap for multi-case handling'

    def __init__(self, swash_proj):
        '''
        initializes wrap

        swash_proj - SwashProject() instance, contains project parameters
        '''

        self.proj = swash_proj              # swash project parameters
        self.io = SwashIO(self.proj)        # swash input/output 
        self.plots = SwashPlot(self.proj)   # swash plotting tool

        # resources
        p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')

        # swan bin executable
        self.bin = op.abspath(op.join(p_res, 'swash_bin', 'swash_ser.exe'))

    def build_cases(self, list_swashinput):
        '''
        generates all files needed for swash multi-case execution

        list_swashinput - list of SwashInput objects
        '''

        # make main project directory
        self.io.make_project()

        # one case for each swash input
        for ix, si in enumerate(list_swashinput):

            # prepare swash input computational grid
            dxL = self.proj.dxL
            d0 = np.abs(self.proj.depth[0])

            si.prepare_computational_grid(dxL, d0, self.proj.b_grid)

            # build case 
            case_id = '{0:04d}'.format(ix)
            self.io.build_case(case_id, si)

    def get_run_folders(self):
        'return sorted list of project cases folders'

        ldir = sorted(os.listdir(self.proj.p_cases))
        fp_ldir = [op.join(self.proj.p_cases, c) for c in ldir]

        # filter only SWASH project folders
        fp_ldir = [p for p in fp_ldir if any(file.endswith('.sws') for file in os.listdir(p))]

        return [p for p in fp_ldir if op.isdir(p)]

    def run_cases(self):
        'run all cases inside project "cases" folder'

        # TODO: improve log / check execution ending status

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        for p_run in run_dirs:

            # run case
            self.run(p_run)

            # log
            p = op.basename(p_run)
            print('SWASH CASE: {0} SOLVED'.format(p))

    def run(self, p_run):
        'Bash execution commands for launching SWASH'

        # aux. func. for launching bash command
        def bash_cmd(str_cmd, out_file=None, err_file=None):
            'Launch bash command using subprocess library'

            _stdout = None
            _stderr = None

            if out_file:
                _stdout = open(out_file, 'w')
            if err_file:
                _stderr = open(err_file, 'w')

            s = sp.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
            s.wait()

            if out_file:
                _stdout.flush()
                _stdout.close()
            if err_file:
                _stderr.flush()
                _stderr.close()

        # check if windows OS
        is_win = sys.platform.startswith('win')

        if is_win:
            # WINDOWS - use swashrun command
            cmd = 'cd {0} && swashrun input && {1} input'.format(
                p_run, self.bin)

        else:
            # LINUX/MAC - ln input file and run swan case
            cmd = 'cd {0} && ln -sf input.sws INPUT && {1} INPUT'.format(
                p_run, self.bin)

            # redirect SWASH output
            cmd += ' 2>&1 > wswash_exec.log'

        bash_cmd(cmd)

    def output_files(self):
        'convert raw SWASH output files to netCDF and store them'

        run_dirs = self.get_run_folders()
        for p_run in run_dirs:
            
            # get case id
            case_id = float(op.basename(p_run))
            
            # run case
            ds = self.io.output_points(p_run, case_id)

            # log
            ds.to_netcdf(op.join(p_run, 'output.nc'))
            print('SWASH CASE: {0} POSTPROCESSED'.format(p_run))


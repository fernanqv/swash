#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.signal import find_peaks

from .spectra import spectral_analysis
from .statistics import upcrossing


class Postprocessor(object):
    'SWASH numerical model customized output postprocessor operations'

    def __init__(self, swash_proj, swash_input, swash_wrap, output_vars, run_post=True):
        '''
        initialize postprocessor

        swash_proj  - SwanProject instance, contains project parameters
        swash_input - SwanInput instance, contains case input parameters
        swash_output - xarray.Dataset, cointains case output (from io.output_points())
        '''

        self.swash_proj = swash_proj
        self.swash_input = swash_input
        self.swash_wrap = swash_wrap

        self.out = None

        # user-specific variables
        self.output_vars = output_vars
        
        # store output in a xarray.Dataset
        if run_post:
            self.ds_output = self.run_vars_block()

        self.ds_output = self.store_vars_block()

    def remove_warmup_nodata(self, out):
        'remove spin up time from every simulation'

        warmup = self.swash_proj.warmup

        # remove warmup time from output
        t = np.where(out['Tsec'].values > warmup)[0]
        out = out.isel(Tsec = t)

        out = out.where(out != -9999, np.nan)

        return out

    def run_vars_block(self):
        '''
        runs postprocessor variables block
        '''
        output_vars = self.output_vars
        p_cases = self.swash_wrap.get_run_folders()

        # Mapeo de variables a funciones de cálculo correspondientes
        var_functions = {
            'Ru2': self.calculate_runup2,
            'RuDist': self.calculate_runup,
            'Msetup': self.calculate_setup,
            'Hrms': self.calculate_statistical_analysis,
            'Hfreqs': self.calculate_spectral_analysis

        }

        for p_case in p_cases:
            
            # read output file
            out = xr.open_dataset(op.join(p_case, 'output.nc'))
            
            # remove warmup time from output
            out = self.remove_warmup_nodata(out)

            self.out = out

            # calculate output variables
            list_ds = []
            for var, func in var_functions.items():
                if var in output_vars:
                    list_ds.append(func())

            # concat individual outputs
            ds_output = xr.merge(list_ds)
            
            # store output in netcdf
            ds_output.to_netcdf(op.join(p_case, 'output_postprocessed.nc'))

    def store_vars_block(self):
        
        'if output has been processed -> store datasets into netcdf'

        p_cases = self.swash_wrap.get_run_folders()

        list_ds = []

        for p_case in p_cases:
            
            out = xr.open_dataset(op.join(p_case, 'output_postprocessed.nc'))

            list_ds.append(out)

        return xr.concat(list_ds, dim='case_id')

    def find_maximas(self, values):
        'find the individual uprushes along the beach profile'

        peaks, _ = find_peaks(values)

        return peaks, values[peaks]

    def calculate_runup2(self):
        '''
        Calculates runup

        returns runup-02
        '''
        
        out = self.out
        case_id = float(out['case_id'].values)

        # get runup
        runup = out['Runlev'].values

        # find individual wave uprushes
        _, val_peaks = self.find_maximas(runup)
        
        # calculate ru2
        
        ru2 = np.percentile(val_peaks, 98)
        
        # create xarray Dataset with ru2 value depending on case_id
        ds = xr.Dataset({'Ru2': ('case_id', [ru2])}, {'case_id': [case_id]})

        return(ds)

    def calculate_runup(self):
        'store complete runup characterization'

        out = self.out

        # get runup
        ds = out['Runlev']

        # create xarray Dataset with ru2 value depending on case_id
        #ds = xr.Dataset({'RuDist': ('Tsec', runup)}, {'Tsec': [case_id]})

        return(ds)


    def calculate_setup(self):
        '''
        Calculate mean set-up

        returns setup xarray.Dataset
        '''
        
        out =  self.out

        # create xarray Dataset with mean setup
        ds = out['Watlev'].mean(dim='Tsec')
        ds = ds.to_dataset()

        # eliminate Yp dimension
        ds = ds.squeeze()

        # rename variable
        ds = ds.rename({'Watlev': 'Msetup'})

        return(ds)

    def calculate_statistical_analysis(self):
        '''
         zero-upcrossing analysis to obtain individual wave heights (Hi) and wave periods (Ti)

        '''
        out = self.out

        # for every X coordinate in domain
        df_Hrms = pd.DataFrame()

        for x in out['Xp'].values:
            
            dsw = out.sel(Xp=x)

            # obtain series of water level
            series_water = dsw['Watlev'].values
            time_series = dsw['Tsec'].values

            # perform statistical analysis 
            #_, Hi = upcrossing(time_series, series_water)
            _, Hi = upcrossing(np.vstack([time_series, series_water]).T)

            # calculate Hrms
            Hrms_x = np.sqrt(np.mean(Hi ** 2))

            df_Hrms.loc[x, 'Hrms'] = Hrms_x

        # convert pd DataFrame to xr Dataset
        df_Hrms.index.name = 'Xp'
        ds = df_Hrms.to_xarray()

        # assign coordinate case_id
        ds = ds.assign_coords({'case_id': [out['case_id'].values]})

        return ds

    def calculate_spectral_analysis(self):
        '''
        makes a water level spectral analysis (scipy.signal.welch)
        then separates incident waves, infragravity waves, very low frequency
        waves.

        returns

        df         - pandas dataframe with analysis output
        ds_fft_hi  -
        '''

        out = self.out
        delttbl = np.diff(out['Tsec'].values)[1]

        df_H_spectral = pd.DataFrame()

        for x in out['Xp'].values:

            dsw = out.sel(Xp=x)
            series_water = dsw['Watlev'].values

            # calculate significant, SS, IG and VLF wave heighs
            Hs, Hss, Hig, Hvlf = spectral_analysis(series_water, delttbl)

            df_H_spectral.loc[x, 'Hs'] = Hs
            df_H_spectral.loc[x, 'Hss'] = Hss
            df_H_spectral.loc[x, 'ig'] = Hig
            df_H_spectral.loc[x, 'Hvlf'] = Hvlf
        
        # convert pd DataFrame to xr Dataset
        df_H_spectral.index.name = 'Xp'
        ds = df_H_spectral.to_xarray()

        # assign coordinate case_id
        ds = ds.assign_coords({'case_id': [out['case_id'].values]})

        return ds

    def calculate_overtopping(self):
        '''
        Calculates overtopping at maximum bathymetry elevation point

        returns acumulated overtopping (l/s/m)
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        dx = self.swash_proj.b_grid.dx
        out = self.swash_output
        warmup = self.swash_input.waves_parameters['warmup']
        tendc = self.swash_input.waves_parameters['tendc']

        # remove warmup time from output
        t = np.where(out['Tsec'].values > warmup)[0]
        outn = out.isel(Tsec = t) 

        # get overtopping at max bathymetry elevation point
        ix_gate_q = int(np.argmax(depth, axis=None, out=None) * dx)
        q = outn.isel(Xp = ix_gate_q).Qmag.values

        # process overtopping nodatavalues and nanvalues
        q[np.where(q == -9999.0)[0]] = np.nan
        q = q[~np.isnan(q)]
        q = q[np.where(q > 0)]

        # acumulated overtopping (units as l/s/m)
        Q = np.nansum(q) * 1000 / tendc

        return Q, q

    def calculate_reflection(self, flume_f=0.25):
        '''
        Calculates waves reflection using Welch's method

        flume_f - fraction of profile length to compute kr

        returns
        Kr - reflection coefficient
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        delttbl = self.swash_proj.delttbl
        H = self.swash_input.waves_parameters['H']
        out = self.swash_output

        # set flume as depth/4 (default flume_f = 0.25)
        flume = int(len(depth) * flume_f)

        # output water level
        sw = out.isel(Xp = flume)['Watlev'].values
        sw = sw[~np.isnan(sw)]

        # estimate power spectral density using Welch's method
        fout, Eout = signal.welch(sw, fs = 1/delttbl , nfft = 512, scaling='density')
        m0out = np.trapz(Eout, x=fout)
        Hsout = 4 * np.sqrt(m0out)

        # calculate reflection coefficient
        Kr = np.sqrt((Hsout/H)-1)

        return Kr
# Import python libraries
# basic
import os
import sys
import os.path as op

# common
import numpy as np
import pandas as pd
import xarray as xr


'''# addpath to bluemath modules
sys.path.insert(0, op.join(os.getcwd(), '..', '..', '..', '..'))

# wrap_swash modules
from wrap_swash.wswash.wrap import SwashProject, SwashWrap, SwashInput
from wrap_swash.wswash.postprocessor import Postprocessor
from wrap_swash.wswash.plots import SwashPlot
from wrap_swash.wswash.io import SwashIO
from wrap_swash.wswash.profiles import reef

# statistical_toolkit  modules:
from scipy.stats import qmc 
from statistical_toolkit.bluemathtk.MDA import *
from statistical_toolkit.bluemathtk.PCA import *
from hyswash.lib.output_extract import *
from hyswash.lib.waves import series_TMA
from hyswash.lib.reconstruction import RBF_Reconstruction_singular, RBF_Reconstruction_spatial'''

from scipy.signal import find_peaks

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

# read output.tab and run.tab to pandas.DataFrame
p_dat = 'output.tab'
p_run = 'run.tab'

ds1 = read_tabfile(p_dat)
ds2 = read_tabfile(p_run)

ds1['Tsec'] = np.round(ds1['Tsec'], 1)
ds2['Tsec'] = np.round(ds2['Tsec'], 1)

# parse pandas.DataFrame to xarray.Dataset
ds1 = ds1.set_index(['Xp', 'Yp','Tsec']) #, coords = Time, Xp, Yp
ds1 = ds1.to_xarray()

ds2 = ds2.set_index(['Tsec']) #, coords = Time, Xp, Yp
ds2 = ds2.to_xarray()

# merge output files to one xarray.Dataset
ds = xr.merge([ds1, ds2], compat='no_conflicts')

current_directory = os.getcwd()
directory_name = os.path.basename(current_directory)
ds.coords['case_id'] = int(directory_name)

def find_maximas(values):
    'find the individual uprushes along the beach profile'
    
    peaks, _ = find_peaks(values)
    
    return peaks, values[peaks]

runup = ds.Runlev.values
_, val_peaks = find_maximas(runup)
ru2 = np.percentile(val_peaks, 98)

# Assuming you have an existing xarray dataset 'ds' and a float value for 'Ru2'
Ru2_value = ru2  # Example value for Ru2

# Create a new DataArray for Ru2
Ru2_array = xr.DataArray(Ru2_value)

# Assign the new variable Ru2 to the dataset
ds_with_Ru2 = ds.assign(Ru2=Ru2_array)

ds_with_Ru2.to_netcdf('output_Ru2.nc')
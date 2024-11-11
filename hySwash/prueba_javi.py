import os
import os.path as op

import sys
sys.path.insert(0, op.join(os.getcwd(),'swash/hySwash/'))

from wrap_swash.wswash.wrap import SwashProject, SwashWrap, SwashIO
from wrap_swash.wswash.postprocessor import Postprocessor
from wrap_swash.wswash.plots import SwashPlot

p_proj = "/home/tausiaj/GitHub-GeoOcean/swash/hySwash/projects_valva/"
n_proj = p_proj + "test_valva/"

# Initialize the project
sp = SwashProject(p_proj, n_proj)
sw = SwashWrap(sp)
si = SwashIO(sp)
sm = SwashPlot(sp)

# Set the simulation period and grid resolution
sp.tendc = 1800                          # simulation period (SEC)
sp.warmup = 0.15 * sp.tendc              # spin-up time (s) (default 15%)

# Postprocess SWASH output (raw to netCDF)
# sw.output_files()

# Define variables to compute
output_vars = ['Ru2', 'Msetup', 'RuDist', 'Hrms', 'Hfreqs']

# Initialize the postprocessor
pp = Postprocessor(sp, si, sw, output_vars=output_vars, run_post=True)
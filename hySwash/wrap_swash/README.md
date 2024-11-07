# Wrap SWASH

A python wrap for SWASH numerical model.


## Table of contents
1. [Description](#desc)
2. [Main Contents](#mc)
3. [Documentation](#doc)
4. [Schemes](#sch)
5. [Install](#ins)
    1. [Install from sources](#ins_src)
    2. [Install SWASH numerical model](#ins_swh)
6. [Examples](#exp)
7. [Contributors](#ctr)
8. [License](#lic)


<a name="desc"></a>
## Description


<a name="mc"></a>
## Main contents

[wswash](./wswash): SWASH numerical model wrapper 
- [io](./wswash/io.py): SWASH numerical model input/output operations
- [plots](./wswash/plots.py): plotting module 
- [postprocessor](./wswash/postprocessor.py): SWASH output postprocessor module
- [profiles](./wswash/profiles.py): customized bathymetry profiles library 
- [spectra](./wswash/spectra.py): spectral analysis for output postprocessing
- [statistics](./wswash/statistics.py): statistical analysis for output postprocessing 
- [waves](./wswash/waves.py): waves series generation library 
- [wrap](./wswash/wrap.py): SWASH numerical model python wrap 


<a name="doc"></a>
## Documentation

SWASH numerical model detailed documentation can be found at: <http://swash.sourceforge.net/>

- [SWASH install/compile manual](http://swash.sourceforge.net/download/download.htm)
- [SWASH user manual](http://swash.sourceforge.net/online_doc/swashuse/swashuse.html)


<a name="sch"></a>
## Schemes

<a name="ins"></a>
## Download Executables
- - -

The last official SWASH Delft execs:  <https://sourceforge.net/projects/swash/files/swash/> 

* Only Windows and Linux executables are available, MacOS users should compile it using the 'tar' file.

<a name="ins"></a>
## Install
- - -

Source code is currently privately hosted on GitLab at:  <https://gitlab.com/geoocean/bluemath/numerical-models-wrappers/wrap_swash> 


<a name="ins_src"></a>
### Install from sources

Install requirements. Navigate to the base root of [wrap\_swash](./) and execute:

```bash
   # Default install, miss some dependencies and functionality
   pip install -r requirements/requirements.txt
```

Then install wrap\_swash:

```bash
   python setup.py install

```


<a name="ins_swh"></a>
### Install SWASH numerical model 

Download and Compile SWASH numerical model:

```bash
  # you may need to install a fortran compiler
  - For Linux:
    sudo apt install gfortran

  - For MacOS:
    brew install gcc

  # download and unpack
  wget https://swash.sourceforge.io/download/zip/swash-7.01AB.tar.gz
  tar -zxvf swash-7.01AB.tar.gz

  # compile numerical model
  cd swash/
  make config
  make ser
```

Copy SWASH binary file to module resources

```bash
  # Launch a python interpreter
  $ python

  Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
  [GCC 8.4.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  
  >>> import wswash 
  >>> wswash.set_swash_binary_file('swash.exe')
```


<a name="exp"></a>
## Examples:

[scripts](./scripts): script examples folder 
- [demo 01](./scripts/demo_01_wavepropagation.py): wave propagation example
- [demo 02](./scripts/demo_02_coupling.py): SWAN output copupling to SWASH example
- [demo 03](./scripts/demo_03_bichromatic.py): bichromatic example 
- [demo 04](./scripts/demo_04_Jonswap.py): JONSWAP spectra example 

[notebooks](./notebooks): notebooks examples folder 
- [SWASH Case](./notebooks/swash_case.ipynb): An easy-to-use Jupyter Notebook to model wave transformation over a shallow cross-shore profile
- [bathymetry examples](./notebooks/bathymetry_profiles_examples.ipynb): examples of customized bathymetry profiles for SWASH case building
- [waves series examples](./notebooks/waves_series_examples.ipynb): examples of wave series generation (monochromatic, bichromatic, Jonswap spectrum)


<a name="ctr"></a>
## Contributors:

Alba Ricondo Cueva (ricondoa@unican.es)\
Fernando Mendez Incera (fernando.mendez@unican.es)


<a name="lic"></a>
## License

This project is licensed under the MIT License - see the [license](./LICENSE.txt) file for details


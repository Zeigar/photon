# README #

This is a Toolbox

# Installation
Install everything in `requirements.txt`:
```commandline
> python3 -m pip install -r requirements.txt
> # or
> conda install --file requirements.txt
```

## FABOLAS
If you want to use FABOLAS you have install the [Eigen-library](http://eigen.tuxfamily.org/index.php?title=Main_Page), [swig](http://www.swig.org/) and [gfortran](https://gcc.gnu.org/wiki/GFortran):
```commandline
> sudo apt install libeigen3-dev swig gfortran
> # or
> yum install libeigen3-dev swig gfortran
```

You also have to install everything in `requirements_fabolas.txt` afterwards:
```commandline
> python3 -m pip install -r requirements_all.txt
> # or
> conda install --file requirements_fabolas.txt
```
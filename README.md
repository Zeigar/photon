# README #

This is a Toolbox

# Installation
Install everything in `requirements.txt`
## FABOLAS
If you want to use FABOLAS you have install the Eigen-library, swig and gfortran:
```commandline
sudo apt-get install libeigen3-dev swig gfortran
```

You also have to install everything in `requirements_fabolas.txt` afterwards:
```commandline
for req in $(cat all_requirements.txt); do python3 -m pip install $req; done
```
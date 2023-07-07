This file consists of some instructions to use the code in this repository and an explanation of the different files and directories. The code fits an interior structure to a planet with/or without a measurement on the love number. 

# Requirements

Follow these steps to install all needed modules:
1. Make sure to first install CEPAM and PyMultinest (see <https://johannesbuchner.github.io/PyMultiNest/install.html>). 

2. Then, type in the command line:


    cd modules
    pip install -r requirements.txt
    pip install . (or 'pip install -e .' if you want to install them in editable mode)

# Description of the files

- clean\_prior\_planet\_new: In here there are directories that are copied to run the model on a new planet. hom/dilute indicates a homogeneous or dilute model, k indicates that the love number is included in the model. In each of these directories there need to be a symbolic link to /CEPAM/data, so if that is not there or not working the symbolic links need to be recreated after CEPAM is installed.
- modules: all python files needed.
- planets\_love\_number.csv: data that is needed to fit an interior model.
- prepare\_run\_planet.py: This is the main file used to run the model. It copies the needed directory from clean\_prior\_planet\_new to a new one and modifies the files lumrun\_planet.py and run_\planet.py to values specified in planets\_love\_number.csv. lumrun\_planet.py will do an evolution run and run\_planet.py will fit a static model using PyMultinest.

 

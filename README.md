# ML pipeline for Solar Dynamics Observatory (SDO) data
This repo contains a configurable pipeline to train ML models on the SDO Dataset described in 
[Galvez et al. (2019, ApJS)](https://iopscience.iop.org/article/10.3847/1538-4365/ab1005) and retrievable from 
 [here](https://github.com/fluxtransport/SDOML).
 
The available models cover two main use-cases:
* learning spatial patterns of the Sun features to arrive at a self-calibration of EUV instruments
* synthesis of one EUV channel from other 3 channels for the design of a AI-enhanced solar telescope

# Publications

The above use cases have been explored in the following publications:

* ["Exploring the Limits of Synthetic Creation of Solar EUV Images via Image-to-Image Translation"](https://iopscience.iop.org/article/10.3847/1538-4357/ac867b/meta)
  ApJ 937 100 (2022) 
 
* ["Multi-Channel Auto-Calibration for the Atmospheric Imaging Assembly using Machine Learning"](https://www.aanda.org/articles/aa/abs/2021/04/aa40051-20/aa40051-20.html)
  A&A 648, A53 (2021)

* ["Auto-Calibration of Remote Sensing Solar Telescopes with Deep Learning"](https://arxiv.org/abs/1911.04008)
   NeurIPS 2019 - ML4PS Workshop

* ["Using U-Nets to Create High-Fidelity Virtual Observations of the Solar Corona"](https://arxiv.org/abs/1911.04006)
  NeurIPS 2019 - ML4PS Workshop

All the results can be reproduced with the code contained in this repo.

The data uncorrected for degradation used in the [autocalibration paper](https://arxiv.org/abs/1911.04008) is 
available [here](https://zenodo.org/record/4430801#.X_xiP-lKhmE).

# How to use the repo

1) Reusable code lives inside src in the form of a package called sdo that can be installed. 
    
    In order to install the package:
    
        1) cd expanding-sdo-capabilities
        2) pip install --user -e .
   
   Please note the core components of this package can be used to design a ML pipeline for use-cases beyond 
   what described above.
        
2) The pipeline to train and test the autocalibration  model can be started by running:
   
        1) export CONFIG_FILE=./config/autocal_paper_config.yaml 
        2) ./src/sdo/main.py -c $CONFIG_FILE 

    it requires access to a SDOML dataset in numpy memory mapped objects format.

3) The pipeline to train and test the virtual telescope model can be started by running:
   
        1) export CONFIG_FILE=./config/virtual_telescope_default.yaml 
        2) ./src/sdo/main.py -c $CONFIG_FILE 

    it requires access to a SDOML dataset in numpy memory mapped objects format.
    
4) Available models can be found in src/models
    
5) Some scripts for data pre-processing are contained in scripts/data_preprocess.
 
6) Notebooks with some analysis of the results live in the folder notebooks.

# More on this project 
This project started as part of the [2019 Frontier Development Lab (FDL) SDO team](https://frontierdevelopmentlab.org/2019-sdo). 
A description of this program is available [here](https://frontierdevelopmentlab.org/about-1).

# Citations
If you decide to reuse this code, please cite [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6954828.svg)](https://doi.org/10.5281/zenodo.6954828)



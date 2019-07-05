# Expanding SDO Capabilities

1) Using spatial information to calibrate the extreme UV (EUV) telescopes: EUV telescopes operating in space are known to degrade over the course of months to years. The rate of degradation is, a priori, unknown. Over the same time scales, the Sun's activity also changes. This project aims to use the spatial patterns of features on the Sun to arrive at a self-calibration of EUV instruments. This would avoid the need to calibrate against other sources.

2) The Solar Dynamics Observatory takes images of the Sun's corona as well as magnetic maps of its surface. The million degree corona exists because of the presence of certain spatial patterns of magnetic fields. In this project we aim to apply style/content transfer techniques to create virtual telescopes. If successful, future NASA missions would be more capable with less hardware. This would allow us to make better space weather forecasts.

We will use the data set described in Galvez et al. (2019, ApJS).
https://iopscience.iop.org/article/10.3847/1538-4365/ab1005

# How to use the repo

1) Notebooks live in the notebooks folder and they follow the following naming convention {number}{initial of your name}_{topic}(i.e. 01v_exploration.ipynb)

2) Reusable code lives inside src in the form of a package called sdo that can be installed. This makes easy to import our codes between notebooks and scripts. 
    In order to install the package:
        1) cd expanding-sdo-capabilities
        2) pip install --user -e .
    In order to import a specific function inside a notebook just use:
        1) from sdo.name_of_the_module import name_of_the_function
        2) Look at notebooks/01v_explore.ipynb for an example

3) There are two directories set up that you can put data and training results into; these directories have been added to `.gitignore` so that they _won't_ be checked into the repo, so that you can safely have training artifacts present without getting git messages on them: `data/` and `training_results/`

# Tutorials/Documents

Documents to help you get up to speed on how we are working as a team:

* IBM Cluster Password-less SSH: https://paper.dropbox.com/doc/IBM-Cluster-Password-less-SSH--AgN6kpiN98HiNFQPzvEiHjZxAQ-cl3lH5Ho6yjBWyZyETqS8
* SDO Git Workflow: https://paper.dropbox.com/doc/SDO-Git-Workflow--AgMf~CdQohUUTtrWKPfTOf1FAQ-fbjyVjGRf7ZHO7d8iHOin
* IBM Cluster: Jupyter Notebooks & Local Editing: https://paper.dropbox.com/doc/IBM-Cluster-Jupyter-Notebooks-Local-Editing--AgOsVInIcJwFv9sNix~GRWNiAQ-rBUYR0tw0kE1l1NPPfsrm


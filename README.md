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


# Tutorials/Documents

Documents to help you get up to speed on how we are working as a team:

* IBM Cluster Password-less SSH: https://paper.dropbox.com/doc/IBM-Cluster-Password-less-SSH--AgN6kpiN98HiNFQPzvEiHjZxAQ-cl3lH5Ho6yjBWyZyETqS8
* SDO Git Workflow: https://paper.dropbox.com/doc/SDO-Git-Workflow--AgMf~CdQohUUTtrWKPfTOf1FAQ-fbjyVjGRf7ZHO7d8iHOin
* IBM Cluster: Jupyter Notebooks & Local Editing: https://paper.dropbox.com/doc/IBM-Cluster-Jupyter-Notebooks-Local-Editing--AgOsVInIcJwFv9sNix~GRWNiAQ-rBUYR0tw0kE1l1NPPfsrm

# Training/Testing Runs

Before you can run the training/testing pipeline, you need to ensure you have your Anaconda and Python PIP environments correctly set up.

SSH into the IBM p10login1 edge host and run the following:

```
ssh p10login1
cd ~/expanding-sdo-capabilities

# Install Anaconda requirements
conda env update -f conda_environment.yml

#Activate the environment
conda activate wmlce_py3_sdo_pipeline

# Install PIP requirements
pip install -r requirements.txt
```

Now you can run the pipeline. Arguments can be passed to `./src/sdo/main.py` either from the command-line as switches, or as a YAML configuration file. run `./src/sdo/main.py --help` to see a list of available configuration options.

Results will be by default saved in a subfolder on a common path: `/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw//experiments_results/`. You should ensure the experiment name is something unique and in order to not risk to overwrite each other it should follow the following convention:
`{number}{your_initials}_experiment_{topic}`. Looks also at the experiment notebook *here*.

To start a new training run:

```
cd ~/expanding-sdo-capabilities
export CONFIG_FILE=config/autocalibration_default.yaml
export EXPERIMENT_NAME=01b_experiment_test
export NUM_EPOCHS=5
./src/sdo/main.py \
    -c $CONFIG_FILE \
    --experiment-name=$EXPERIMENT_NAME \
    --num-epochs=$NUM_EPOCHS
```

Where `CONFIG_FILE` is a path to a YAML file that might have common configuration options
that you don't want to have to type every time on the command line (see the above
`config/autocalibration_default.yaml` for an example); `EXPERIMENT_NAME` is a unique
experiment name used to partition your training results to 
`/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw//experiments_results/$EXPERIMENT_NAME`;
and NUM_EPOCHS is the total number of training epochs you want.

To resume a previously checkpointed training session:

```
cd ~/expanding-sdo-capabilities
export CONFIG_FILE=config/autocalibration_default.yaml
export EXPERIMENT_NAME=01b_experiment_1
export START_EPOCH_AT=2
export NUM_EPOCHS=5
export RESULTS_PATH=/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw//experiments_results
./src/sdo/main.py \
    -c $CONFIG_FILE \
    --experiment-name=$EXPERIMENT_NAME \
    --num-epochs=$NUM_EPOCHS \
    --continue-training=True \
    --saved-model-path=$RESULTS_PATH/$EXPERIMENT_NAME/model_epoch_$START_EPOCH_AT.pth \
    --saved-optimizer-path=$RESULTS_PATH/$EXPERIMENT_NAME/optimizer_epoch_$START_EPOCH_AT.pth \
    --start-epoch-at=$START_EPOCH_AT
```

Where `START_EPOCH_AT` is the new training epoch to begin training from.

Note that both in the YAML config file and on the command line, the major pipeline to run
(whether the autocalibration architecture or the encoder/decoder architecture), is controlled
by `--pipeline-name`, which can either be `AutocalibrationPipeline` or `EncoderDecoderPipeline`.
`EncoderDecoderPipeline` is not yet implemented.

To easily copy over training artifacts from a run to see how things went, first add the following
to your laptop's `~/.bash_profile` or `~/.bashrc` file:

```
sync_results_func() {
        rsync --protocol=28 -vrzhe ssh --progress --exclude '.git' --exclude .DS_Store --exclude *.pth p10login1:/gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/experiments_results$1 experiments_results
}
alias sync_results=sync_results_func
```

Quit and save, then:

```
source ~/.bash_profile
```

Now you can use the following command to easily pull results back over to your laptop to view them:

```
export EXPERIMENT_NAME=01b_experiment_1
cd ~/expanding-sdo-capabilities
sync_results
open ./experiments_results/$EXPERIMENT_NAME
```

Note that this skips syncing the very large `*.pth` files for saved checkpoint models and optimizer
details to your laptop; those will remain on the IBM machine.

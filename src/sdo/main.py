#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import sys
import logging

import configargparse
from configargparse import YAMLConfigFileParser

import torch

from sdo import __version__
from sdo.pipelines.autocalibration_pipeline import AutocalibrationPipeline
from sdo.py_utils import split_path
from sdo.pytorch_utilities import init_gpu, set_seed


__author__ = "vale-salvatelli"
__copyright__ = "vale-salvatelli"
__license__ = "mit"

_logger = logging.getLogger(__name__)

"""
Provides a common pipeline for running and restarting training/testing experiments.

Arguments can be passed to main.py either from the command-line as switches, or
as a YAML configuration file. run `./src/sdo/main.py --help` to see a list of
available configuration options.


To start a new training run:

cd ~/expanding-sdo-capabilities
export CONFIG_FILE=config/autocalibration_default.yaml
export EXPERIMENT_NAME=01b_experiment_1
export NUM_EPOCHS=5
./src/sdo/main.py \
    -c $CONFIG_FILE \
    --experiment-name=$EXPERIMENT_NAME \
    --num-epochs=$NUM_EPOCHS

Where CONFIG_FILE is a path to a YAML file that might have common configuration options
that you don't want to have to type every time on the command line (see the above
config/autocalibration_default.yaml for an example); EXPERIMENT_NAME is a unique
experiment name used to partition your training results to ./training_results/$EXPERIMENT_NAME;
and NUM_EPOCHS is the total number of training epochs you want.


To resume a previously checkpointed training session:

cd ~/expanding-sdo-capabilities
export CONFIG_FILE=config/autocalibration_default.yaml
export EXPERIMENT_NAME=01b_experiment_1
export START_EPOCH_AT=2
export NUM_EPOCHS=5
./src/sdo/main.py \
    -c $CONFIG_FILE \
    --experiment-name=$EXPERIMENT_NAME \
    --num-epochs=$NUM_EPOCHS \
    --continue-training=True \
    --saved-model-path=./training_results/$EXPERIMENT_NAME/model_epoch_$START_EPOCH_AT.pth \
    --saved-optimizer-path=./training_results/$EXPERIMENT_NAME/optimizer_epoch_$START_EPOCH_AT.pth \
    --start-epoch-at=$START_EPOCH_AT

Where START_EPOCH_AT is the new training epoch to begin training from.

Note that both in the YAML config file and on the command line, the major pipeline to run
(whether the autocalibration architecture or the encoder/decoder architecture), is controlled
by --pipeline-name, which can either be AutocalibrationPipeline or EncoderDecoderPipeline.
EncoderDecoderPipeline is not yet implemented.

To easily copy over training artifacts from a run to see how things went, first add the following
to your laptop's ~/.bash_profile or ~/.bashrc file:

sync_results_func() {
        rsync -vrzhe ssh --progress --exclude '.git' --exclude .DS_Store --exclude *.pth p10login1:~/expanding-sdo-capabilities/training_results/$1 training_results
}
alias sync_results=sync_results_func

Quit and save, then:

source ~/.bash_profile

Now you can use the following command to easily pull results back over to your laptop to view them:

export EXPERIMENT_NAME=01b_experiment_1
cd ~/expanding-sdo-capabilities
sync_results
open ./training_results/$EXPERIMENT_NAME

Note that this skips syncing the very large *.pth files for saved checkpoint models and optimizer
details to your laptop; those will remain on the IBM machine.
"""
def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Path is assumed to be relative to being inside of the
    # git repo expanding-sdo-capabilities directory.
    p = configargparse.ArgParser(config_file_parser_class=YAMLConfigFileParser,
                                 default_config_files=['./configs/*.conf'],
                                 description='Training/testing pipeline')
    p.add_argument(
        '--version',
        action='version',
        version='sdo {ver}'.format(ver=__version__))
    p.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        default=logging.INFO,
        action='store_const',
        const=logging.INFO)
    p.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    p.add_argument(
        '--log-minimal',
        dest='log_minimal',
        default=False,
        type=bool,
        help="If true, log lines will have no prefix; if false, extensive prefix logging will appear")
    p.add_argument(
        '-c',
        '--config',
        is_config_file=True,
        help='Config file path with YAML values instead of command line switches')
    p.add_argument(
        '-p',
        '--pipeline-name',
        required=True,
        dest='pipeline_name',
        help='Which pipeline to use: AutocalibrationPipeline or EncoderDecoderPipeline')
    p.add_argument(
        '--experiment-name',
        dest='experiment_name',
        default=None,
        help='The name of this experiment, used to partition result artifacts; defaults to date and time')
    p.add_argument(
        '--results-path',
        dest='results_path',
        default='./training_results',
        help='Where to store generated logs, models, etc., relative to --artifacts-root-path',
        )
    p.add_argument(
        '--num-epochs',
        dest='num_epochs',
        type=int,
        default=5,
        help='Number of training epochs'
        )
    p.add_argument(
        '--batch-size-train',
        dest='batch_size_train',
        type=int,
        default=64,
        help='Batch size for training'
        )
    p.add_argument(
        '--batch-size-test',
        dest='batch_size_test',
        type=int,
        default=100,
        help='Batch size for testing')
    p.add_argument(
        '--save-interval',
        dest='save_interval',
        type=int,
        default=5,
        help='Every save-interval epochs we will save the trained model and optimizer state')
    p.add_argument(
        '--log-interval',
        dest='log_interval',
        type=int,
        default=10,
        help='While processing batches during training, how often to print out log statistics')
    p.add_argument(
        '--wavelengths',
        dest='wavelengths',
        nargs='+',
        default='0094 0131',
        help='Wavelengths to use for input; Ex: --wavelengths 0094 0131')
    p.add_argument(
        '--instruments',
        dest='instruments',
        nargs='+',
        default='AIA AIA',
        help='For each input wavelength, the instrument to use for its data; Ex: --instruments AIA AIA')
    p.add_argument(
        '--num-channels',
        dest='num_channels',
        type=int,
        default=2,
        help='The number of input channels for the deep net')
    p.add_argument(
        '--actual-resolution',
        dest='actual_resolution',
        type=int,
        default=512,
        help='Actual pixel resolution of training/testing images before subsampling is applied')
    p.add_argument(
        '--subsample',
        dest='subsample',
        type=int,
        default=4,
        help='Indicates what to reduce images by, against --actual-resolution. Ex: 512/4 = 128')
    p.add_argument(
        '--cuda-device',
        dest='cuda_device',
        type=int,
        default=None,
        help='CUDA GPU device number to use; if not provided a random CUDA GPU on the system will be used')
    p.add_argument(
        '--random-seed',
        dest='random_seed',
        type=int,
        default=1,
        help='Random seed to use for initializing sources of randomness')
    p.add_argument(
        '--deterministic-cuda',
        dest='determininistic_cuda',
        type=bool,
        default=True,
        help='Whether to force CUDA to be deterministic; can cause some perf slowdown')
    p.add_argument(
        '--continue-training',
        dest='continue_training',
        type=bool,
        default=False,
        help='Whether to continue training from a saved checkpoint')
    p.add_argument(
        '--saved-model-path',
        dest='saved_model_path',
        help='Path to a saved model to continue training from')
    p.add_argument(
        '--saved-optimizer-path',
        dest='saved_optimizer_path',
        help='Path to a saved optimizer to continue training from')
    p.add_argument(
        '--start-epoch-at',
        dest='start_epoch_at',
        type=int,
        default=1,
        help='When restarting training, the epoch to start at')
    p.add_argument(
        '--yr-range',
        nargs='+',
        type=int,
        default=[],
        help='Start and stop year range, inclusive at both ends. Ex: --yr-range 2012 2013')

    args = vars(p.parse_args(args))

    if not args['experiment_name']:
        args['experiment_name'] = f"experiment-{datetime.datetime.now():%Y-%m-%d-time-%H-%M-%S}"

    # Make downstream processing easier by expanding paths.
    args['results_path'] = os.path.abspath(os.path.join(args['results_path'],
                                           args['experiment_name']))

    if args['continue_training']:
        if not args['saved_model_path'] or not args['saved_optimizer_path']:
            raise Exception('To continue training you must provide: --saved-model-path, '
                            '--saved-optimizer-path, and --start-epoch-at')
        args['saved_model_path'] = os.path.abspath(args['saved_model_path'])
        args['saved_optimizer_path'] = os.path.abspath(args['saved_optimizer_path'])

    args['scaled_width'] = int(args['actual_resolution'] / args['subsample'])
    args['scaled_height'] = args['scaled_width']
    _logger.info('Actual resolution: {}, subsample: {}, scaled size: {}',
        args['actual_resolution'], args['subsample'], args['scaled_width'])

    return configargparse.Namespace(**args)
    

def setup_logging(loglevel, minimal):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "%(message)s" if minimal else "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def create_dirs(args):
    """Create any missing directories needed for training/testing artifacts."""
    path = os.path.abspath(args.results_path)
    if not os.path.exists(path):
        _logger.info('{} does not exist; creating directory...'.format(path))
        os.makedirs(path)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel, args.log_minimal)
    _logger.info('Parsed arguments: {}'.format(args))

    create_dirs(args)

    device = init_gpu(args.cuda_device)
    set_seed(args.random_seed, args.determininistic_cuda)

    pipeline = None
    _logger.info('Using {}'.format(args.pipeline_name))
    if args.pipeline_name == 'AutocalibrationPipeline':
        pipeline = AutocalibrationPipeline(num_channels=args.num_channels,
                                           scaled_height=args.scaled_height,
                                           scaled_width=args.scaled_width,
                                           device=device,
                                           instruments=args.instruments,
                                           wavelengths=args.wavelengths,
                                           subsample=args.subsample,
                                           batch_size_train=args.batch_size_train,
                                           batch_size_test=args.batch_size_test,
                                           log_interval=args.log_interval,
                                           results_path=args.results_path,
                                           num_epochs=args.num_epochs,
                                           save_interval=args.save_interval,
                                           continue_training=args.continue_training,
                                           saved_model_path=args.saved_model_path,
                                           saved_optimizer_path=args.saved_optimizer_path,
                                           start_epoch_at=args.start_epoch_at,
                                           yr_range=args.yr_range)
    elif args.pipeline_name == 'EncoderDecoderPipeline':
        raise Exception('EncoderDecoderPipeline not implemented yet!')
        # TODO: Implement
    else:
        raise Exception('Unknown pipeline: {}'.format(args.pipeline_name))

    # Start training and testing in the selected pipeline.
    pipeline.run()


def run():
    """ Entry point for console_scripts """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

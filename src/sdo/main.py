#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import yaml

import torch

from sdo.io import format_epoch
from sdo.parse_args import parse_args
from sdo.pytorch_utilities import init_gpu, set_seed
from sdo.pipelines.autocalibration_pipeline import AutocalibrationPipeline


__author__ = "vale-salvatelli"
__copyright__ = "vale-salvatelli"
__license__ = "mit"

_logger = logging.getLogger(__name__)


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
        os.makedirs(path, mode=0o755)


def save_config_details(args, results_path, experiment_name):
    """
    Given some final arguments, save them as a YAML config to make it easier to
    both see what was configured as well as easily re-run an experiment given some config.
    """
    args_dict = vars(args)
    filename = os.path.join(results_path, '{}_config_{}.yaml'.format(
        format_epoch(0), experiment_name))
    with open(filename, 'w') as outfile:
        yaml.dump(args_dict, outfile, default_flow_style=False)
    _logger.info('Saved final YAML config details to {}'.format(filename))


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel, args.log_minimal)
    _logger.info('Parsed arguments: {}'.format(args))

    create_dirs(args)
    save_config_details(args, args.results_path, args.experiment_name)

    device = init_gpu(args.cuda_device)
    set_seed(args.random_seed, args.determininistic_cuda)

    pipeline = None
    _logger.info('Using {}'.format(args.pipeline_name))
    if args.pipeline_name == 'AutocalibrationPipeline':
        pipeline = AutocalibrationPipeline(exp_name=args.experiment_name,
                                           model_version=args.model_version,
                                           actual_resolution=args.actual_resolution,
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
                                           yr_range=args.yr_range,
                                           mnt_step=args.mnt_step,
                                           day_step=args.day_step,
                                           h_step=args.h_step,
                                           min_step=args.min_step,
                                           dataloader_workers=args.dataloader_workers,
                                           scaling=args.scaling,
                                           normalization=args.normalization_by_max,
                                           return_random_dim=args.return_random_dim)
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

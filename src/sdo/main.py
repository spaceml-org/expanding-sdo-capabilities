#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configargparse
import datetime
import os
import sys
import logging

from sdo import __version__
from py_utils import split_path


__author__ = "vale-salvatelli"
__copyright__ = "vale-salvatelli"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Path is assumed to be relative to being inside of the
    # git repo expanding-sdo-capabilities directory.
    p = configargparse.ArgParser(default_config_files=['./configs/*.conf'],
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
        default=True,
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
        '-c',
        '--my-config',
        is_config_file=True,
        help='Config file path with YAML values instead of command line switches')
    p.add_argument(
        '--artifacts-root-path',
        dest='artifacts_root_path',
        default='.',
        help='Root path for where to store generated training/testing artifacts',
        )
    p.add_argument(
        '--experiment-name',
        dest='experiment_name',
        default=None,
        help='The name of this experiment, used to partition result artifacts; defaults to date and time')
    p.add_argument(
        '--results-path',
        dest='results_path',
        default='training_results',
        help='Where to store generated logs, models, etc., relative to --artifacts-root-path',
        )
    p.add_argument(
        '--model-filename',
        dest='model_filename',
        default='model.pth',
        help='The filename of the saved generated model, relative to --results-path',
        )
    p.add_argument(
        '--optimizer-filename',
        dest='optimizer_filename',
        default='optimizer.pth',
        help='The filename of saved optimizer details, relative to --results-path',
        )
    p.add_argument(
        '--num-epochs',
        dest='num_epochs',
        default=5,
        help='Number of training epochs'
        )
    p.add_argument(
        '--batch-size-train',
        dest='batch_size_train',
        default=64,
        help='Batch size for training'
        )
    p.add_argument(
        '--batch-size-test',
        dest='batch_size_test',
        default=100,
        help='Batch size for testing')
    p.add_argument(
        '--log-interval',
        dest='log_interval',
        default=10,
        help='While processing batches during training, how often to print out log statistics')
    p.add_argument(
        '--height',
        dest='height',
        default=128,
        help='Pixel height of images for training and testing')
    p.add_argument(
        '--width',
        dest='width',
        default=128,
        help='Pixel width of images for training and testing')
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
        default=2,
        help='The number of input channels for the deep net')
    p.add_argument(
        '--subsample',
        dest='subsample',
        default=4,
        help='By default, images in the SDO dataset are 512x512; subsample indicates what to reduce them by. Ex: 512/4 = 128')

    args = vars(p.parse_args(args))

    if not args['experiment_name']:
        args['experiment_name'] = f"experiment-{datetime.datetime.now():%Y-%m-%d-time-%H-%M-%S}"

    # Expand some paths and values to make downstream processing simpler.
    args['results_path'] = os.path.join(args['artifacts_root_path'],
                                        args['results_path'],
                                        args['experiment_name'])
    args['model_filename'] = os.path.join(args['results_path'], args['model_filename'])
    args['optimizer_filename'] = os.path.join(args['results_path'], args['optimizer_filename'])

    # ConfigArgParse doesn't consistently treat lists in YAML and the command-line.
    if isinstance(args['wavelengths'], list):
        assert len(args['wavelengths']) == 1, 'The behavior of ConfigArgParse has changed'
        args['wavelengths'] = args['wavelengths'][0]
    if isinstance(args['instruments'], list):
        assert len(args['instruments']) == 1, 'The behavior of ConfigArgParse has changed'
        args['instruments'] = args['instruments'][0]

    args['wavelengths'] = args['wavelengths'].split(' ')
    args['instruments'] = args['instruments'].split(' ')

    return configargparse.Namespace(**args)
    

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def create_dirs(args):
    """Create any missing directories."""
    path = os.path.abspath(args.results_path)
    if not os.path.exists(path):
        logging.info('{} does not exist; creating directory...'.format(path))
        os.makedirs(path)

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    logging.info('Parsed arguments: {}'.format(args))

    create_dirs(args)

    # _logger.debug("Starting crazy calculations...")
    # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    # _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

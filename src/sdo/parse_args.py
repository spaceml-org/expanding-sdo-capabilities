import datetime
import logging
import os

import configargparse
from configargparse import YAMLConfigFileParser

from sdo import __version__
from sdo.global_vars import EXP_RESULTS_BASEDIR


_logger = logging.getLogger(__name__)


"""
Provides a common pipeline for running and restarting training/testing experiments.
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
        '--model-version',
        dest='model_version',
        type=int,
        required=True,
        help='Which version of the model for your particular pipeline you want to run')
    p.add_argument(
        '--results-path',
        dest='results_path',
        default=EXP_RESULTS_BASEDIR,
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
        default=50,
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
        help='Absolute path to a saved model to continue training from')
    p.add_argument(
        '--saved-optimizer-path',
        dest='saved_optimizer_path',
        help='Absolute path to a saved optimizer to continue training from')
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
    p.add_argument(
        '--mnt-step',
        dest='mnt_step',
        type=int,
        default=1,
        help='Month frequency, starting from January. Values must be 1 or greater.')
    p.add_argument(
        '--day-step',
        dest='day_step',
        type=int,
        default=1,
        help='Day frequency starting from 1. Values must be 1 or greater.')
    p.add_argument(
        '--h-step',
        dest='h_step',
        type=int,
        default=6,
        help='Hourly frequency starting from 0. Values must be 1 or greater.')
    p.add_argument(
        '--min-step',
        dest='min_step',
        type=int,
        default=60,
        help='Minute frequency starting from 0. Values must be 1 or greater.')
    p.add_argument(
        '--dataloader-workers',
        dest='dataloader_workers',
        type=int,
        default=6,
        help='The number of workers to use when preparing data for feeding into the deep net')
    p.add_argument(
        '--scaling',
        dest='scaling',
        type=bool,
        default=False,
        help='If True scaling of the images by mean of the channel is applied. Look at the values'
             'inside sdo_dataset.py for more detail.')
    p.add_argument(
        '--norm-by-orig-img-max',
        dest='norm_by_orig_img_max',
        type=bool,
        default=False,
        help='If True, dimmed images are normalized by the _original_ image max value.')
    p.add_argument(
        '--norm-by-dimmed-img-max',
        dest='norm_by_dimmed_img_max',
        type=bool,
        default=True,
        help='If True, dimmed images are normalized by their _own_ max value.')
    p.add_argument(
        '--return-random-dim',
        dest='return_random_dim',
        type=bool,
        default=False,
        help='If True, return fake random numbers for the brightness dimming factors during training')

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
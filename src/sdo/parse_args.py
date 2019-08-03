import logging
import os
import pprint

import configargparse
from configargparse import YAMLConfigFileParser

from sdo import __version__


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
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="If true, log lines will have no prefix; if false, extensive prefix logging will appear")
    p.add_argument(
        '-c',
        '--config',
        is_config_file=True,
        help='Config file path with YAML values instead of command line switches')
    p.add_argument(
        '-p',
        '--pipeline-name',
        dest='pipeline_name',
        type=str,
        required=True,
        help='Which pipeline to use: AutocalibrationPipeline or EncoderDecoderPipeline')
    p.add_argument(
        '--experiment-name',
        dest='experiment_name',
        type=str,
        required=True,
        help='The name of this experiment, used to partition result artifacts')
    p.add_argument(
        '--model-version',
        dest='model_version',
        type=int,
        required=True,
        help='Which version of the model for your particular pipeline you want to run')
    p.add_argument(
        '--results-path',
        dest='results_path',
        required=True,
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
        '--test-ratio',
        dest='test_ratio',
        type=float,
        default=0.3,
        help='What percentage of the data to retain for testing')
    p.add_argument(
        '--save-interval',
        dest='save_interval',
        type=int,
        default=50,
        help='Every save-interval epochs we will save the trained model and optimizer state')
    p.add_argument(
        '--add-metrics-interval',
        dest='additional_metrics_interval',
        type=int,
        default=5,
        help='Every additional_metrics_interval epochs we will save the additional metrics')
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
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to force CUDA to be deterministic; can cause some perf slowdown')
    p.add_argument(
        '--continue-training',
        dest='continue_training',
        type=str2bool,
        nargs='?',
        const=True,
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
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='If True scaling of the images by mean of the channel is applied. Look at the values'
             'inside sdo_dataset.py for more detail.')
    p.add_argument(
        '--norm-by-orig-img-max',
        dest='norm_by_orig_img_max',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='If True, dimmed images are normalized by the _original_ image max value.')
    p.add_argument(
        '--norm-by-dimmed-img-max',
        dest='norm_by_dimmed_img_max',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='If True, dimmed images are normalized by their _own_ max value.')
    p.add_argument(
        '--tolerance',
        dest='tolerance',
        type=float,
        default=0.05,
        help='Maximum absolute difference between predicted and ground truth value of the dimming factors.'
             'If the difference is above this value the prediction is considered unsuccessful.'
             'This value affects the computation of the primary metric (frequency of binary success).')
    p.add_argument(
        '--return-random-dim',
        dest='return_random_dim',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='If True, return fake random numbers for the brightness dimming factors during training')
    p.add_argument(
        '--optimizer-weight-decay',
        type=float,
        default=0,
        help='The weight decay to use for whatever optimizer might be used; current default Torchs Adam default')
    p.add_argument(
        '--optimizer-lr',
        type=float,
        default=1e-3,
        help='The learning rate to use for whatever optimizer might be used; current default Torchs Adam default')
    p.add_argument(
        '--min-alpha',
        dest='min_alpha',
        type=float,
        default=0.01,
        help='Smaller degradation factor that can be randomly generated. The maximum is currently fixed to 1.')


    args = vars(p.parse_args(args))

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
    
    # The logger is not setup yet here, as we need the configargs themselves
    # to configure it, so we just print to standard out these details.
    print('\nParsed configuration:\n\n{}'.format(pprint.pformat(args, indent=2)))
    return configargparse.Namespace(**args)


def str2bool(v):
    """
    Boolean ArgParse options are confusing by default: They just just --some-option
    _without_ giving a True or False value; if the switch is present, such as
    --some-option, then it becomes True. If someone gives '--some-option False'
    the option will still evaluate as True! This method changes this default
    behavior so that boolean command line args match expected behavior:
    '--some-option=True' will evaluate to True and '--some-option=False'
    will evaluate to False.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')
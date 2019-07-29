import sys
import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf
import PIL.Image

from sdo.io import sdo_find, sdo_bytescale
from scipy.misc import bytescale
from dataset_tool import TFRecordExporter, error


def create_from_sdo(tfrecord_dir, image_dir, shuffle=True):
    initial_size = 512
    instrs = ['AIA', 'AIA', 'HMI']
    channels = ['0171', '0193', 'bz']
    print('Loading SDOML from "%s"' % image_dir)
    files = []
    # TODO convert into command line params
    for y in np.arange(2010, 2015):
        for m in np.arange(1, 13):
            for d in np.arange(1, 32):
                for h in np.arange(0, 24, 6):
                    result = sdo_find(
                        y, m, d, h, 0,
                        initial_size=initial_size,
                        instrs=instrs,
                        channels=channels,
                        basedir=image_dir)
                    if result != -1:
                        files.append(result)
    print(files)
    print("Number of SDO files = ", len(files))
    if len(files) == 0:
        error('No input images found')

    #img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = 512  # img.shape[0]
    subsample = 1

    # if img.shape[1] != resolution:
    #    error('Input images must have the same width and height')
    # if resolution != 2 ** int(np.floor(np.log2(resolution))):
    #    error('Input image resolution must be a power-of-two')
    # if channels not in [1, 3]:
    #    error('Input images must be stored as RGB or grayscale')
    
    with TFRecordExporter(tfrecord_dir, len(files)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(files))
        for idx in range(order.size):
            img = np.zeros(shape=(int(resolution/subsample),
                                  int(resolution/subsample), 3), dtype=np.uint8)
            img[:, :, 0] = sdo_bytescale(np.load(files[order[idx]][0])[
                                               'x'][::subsample, ::subsample], channels[0])
            img[:, :, 1] = sdo_bytescale(np.load(files[order[idx]][1])[
                                               'x'][::subsample, ::subsample], channels[1])
            img[:, :, 2] = sdo_bytescale(np.load(files[order[idx]][2])[
                                               'x'][::subsample, ::subsample], channels[2])
            #img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose([2, 0, 1])  # HWC => CHW
            tfr.add_image(img)
            
#----------------------------------------------------------------------------
def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)
    
    p = add_command(    'create_from_sdo', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)
    
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------

#!/bin/bash
# This script turns our data from npz files to mm files, which are much faster
# to work with. It is designed to run on Google Cloud Platform. To use, make sure
# you have mounted a large hard drive on your instance (~10TBs), mounted at
# /gpfs/gpfs_gl4_16mb/b9p111/. It's also recommended that you use an instance
# with 12 virtual cores. This script will run very slowly, so make sure to
# open a tmux session before you run it.

# Exit when any command fails.
set -e

process_year () {
    mkdir -p /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML/$1
    mkdir -p /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOMLmm/$1
    gsutil -m cp -r -n gs://fdl-sdo-data/$1 /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML
    cd /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML/$1
    for f in *.tar; do echo "$f" && tar --skip-old-files -xzf "$f"; done
    cd ~/expanding-sdo-capabilities/scripts/data_preprocess
    python convert_to_mm.py -y $1
    gsutil -m cp -r -n /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOMLmm/$1 gs://fdl-sdo-data/SDOMLmm/
    rm -fr /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOML/$1
    rm -fr /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOMLmm/$1 
}

process_year 2010
process_year 2011
process_year 2012
process_year 2013
process_year 2014
process_year 2015
process_year 2016
process_year 2017
process_year 2018

# Generate the inventory
gsutil -m cp -r gs://fdl-sdo-data/SDOMLmm /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/
cd ~/expanding-sdo-capabilities/scripts/data_preprocess
python create_mm_inventory.py
gsutil cp /gpfs/gpfs_gl4_16mb/b9p111/fdl_sw/SDOMLmm/inventory.pkl gs://fdl-sdo-data/SDOMLmm/inventory.pkl
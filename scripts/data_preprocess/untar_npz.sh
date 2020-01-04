#!/bin/bash
# Script that can be run on multiple GCP instances to download the tar for a year given in $YEAR,
# untar it, then upload the results back into GCP storage buckets for the NPZ files. Start
# this script on different instances and change the YEAR variable below to the year to convert.
set -e

export YEAR=2011
export DEVICE=sdb
export ROOT=/mnt/disks/$DEVICE
export DEST=gs://fdl-sdo-data/SDOMLnpz

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/$DEVICE
sudo mkdir -p $ROOT
sudo mount -o discard,defaults /dev/$DEVICE $ROOT

sudo chown -R $(whoami) $ROOT
mkdir -p $ROOT/npz/$YEAR
gsutil -m cp -r gs://fdl-sdo-data/$YEAR $ROOT/npz

cd $ROOT/npz/$YEAR
for file in *.tar
do
  tar xvf "${file}" && rm "${file}"
  ls | grep -v '\.tar$' | xargs -i gsutil -m cp -r {} $DEST/$YEAR/
  ls | grep -v '\.tar$' | xargs -i rm -fr {}
done

echo FINISHED!

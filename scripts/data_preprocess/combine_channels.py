#!/usr/bin/env python
"""
Script that will merge all the channel NPZ files together for a single time step on Google's GCP
storage buckets to efficiently fetch a single time step in one go.

Spin up a host with alot of CPUs. Each host will process a single year. Change the
INSTANCE_NAME below:

export YEAR=2010
export INSTANCE_NAME="process-$YEAR"
export ZONE="us-west1-a"
export INSTANCE_TYPE="n1-standard-32"
export IMAGE="fdl-sdo-2019-image-1"
export SERVICE_ACCOUNT="524456442905-compute@developer.gserviceaccount.com"
gcloud compute instances create $INSTANCE_NAME \
  --zone $ZONE \
  --image $IMAGE \
  --maintenance-policy TERMINATE \
  --machine-type $INSTANCE_TYPE \
  --boot-disk-size 200GB \
  --service-account $SERVICE_ACCOUNT \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/compute

Sync the files over from your laptop to the remote host:
cd ~/src/fdl/expanding-sdo-capabilities
googsyncsdo . $INSTANCE_NAME

On the remote GCP host run the script and specify a year to process:

export YEAR=2010
time python ./scripts/data_preprocess/combine_channels.py --year $YEAR

Once the script is run, run it one more time to process any files that were not successful. The script is smart
enough to not re-generate things that already exist, so this step will finish quickly:

time python ./scripts/data_preprocess/combine_channels.py --year $YEAR

Run the above commands for all years (2010-2018) across multiple machines.

Finally, when finished, you should re-run create_inventory.py to re-generate an
inventory.pkl file has the 'all' channel included for each timestamp.

If you run into permission problems: Make sure the project wide service account has the right cloud
storage roles. Run this on your local laptop:
export SERVICE_ACCOUNT=524456442905-compute@developer.gserviceaccount.com
export BUCKET_NAME=fdl-sdo-data
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:roles/storage.objectAdmin gs://$BUCKET_NAME
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:roles/storage.objectCreator gs://$BUCKET_NAME
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:roles/storage.objectViewer gs://$BUCKET_NAME

Note that you must also have the correct GCP access scopes set for your service provider
account for this script to work:
* Compute Engine - Read Write
* Storage - Full

=======

Example of how you'd reload one of these compressed all-in-one channel files after this script
is finished pre-processing everything:

client = storage.Client()
bucket = client.get_bucket('fdl-sdo-data')
filename = 'SDOMLnpz/2010/05/01/HMI20100501_0036_all.pklz'
blob = bucket.get_blob(filename)
results = pickle.loads(bz2.decompress(blob.download_as_string()))

'results' is an array of tuples, where each tuple contains the string of the channel name,
such as 'bx', with the numpy array of its image results at 512x512 resolution.
"""


from argparse import ArgumentParser
import bz2
import concurrent.futures
import itertools
from io import BytesIO
import multiprocessing as mp
import re
import os
import pickle
import time

from google.auth import compute_engine
from google.cloud import storage

import numpy as np
import pandas as pd


# The number of processes we want to spawn in parallel to handle converting timestamp chunks.
NUM_CONSUMERS = 31

# The number of timestamp channel groups we want to download and convert in one shot.
NUM_TIMESTAMPS_AT_ONCE_PER_PROCESS = 10

# Overall GCP project name.
GCP_PROJECT_NAME = 'space-weather-sdo'

# Bucket name of where we store our SDO data.
GCP_BUCKET_NAME = 'fdl-sdo-data'

# Path to NPZ inventory for quick look ups.
GCP_INVENTORY_PATH = 'SDOMLnpz/inventory.pkl'

# Set to a number like 20 to test processing a subset of data during testing.
#MAX_ITER = 20
MAX_ITER = None

# Lock to safely have threads and processes print to the screen.
global console_lock
console_lock = mp.Lock()


def safeprint(msg):
    """
    Make sure multiple processes don't step on each others feet when printing out to the
    console.
    """
    with console_lock:
        print(msg, flush=True)

        
def enumerate_timestamp_channels(df, year, max_iter):
    """
    Group each of the timestamps into all of their channels, returning
    the results chunked together to handle as a group of channels per
    timestamp. max_iter is useful for debugging to limit the number of
    results we iterate and produce.
    """
    indexes = ['year', 'month', 'day', 'hour', 'min']
    counter = 0
    for _, timestamp_group in df[df['year'] == year].groupby(indexes):
        channels = [
            {'channel': row['channel'], 'file': row['file']}
            for _, row in timestamp_group.iterrows() if '.pklz' not in row['file']
        ]
        yield channels
        
        counter += 1
        if max_iter is not None and counter >= max_iter:
            break

            
def chunked_iterable(iterable, chunk_size):
    """
    Iterate through something in 'chunks', where each chunk is returned as a group.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

        
def connect_gcp(gcp_bucket_name, gcp_project_name):
    """
    Connect to the Google Cloud Provider storage bucket.
    """
    credentials = compute_engine.Credentials()
    client = storage.Client(credentials=credentials, project=GCP_PROJECT_NAME)
    bucket = client.get_bucket(gcp_bucket_name)
    return client, bucket


def get_combined_path(channel_path):
    """
    Given a path to a channel, generates an *_all.pklz version of that path pointing
    to the combined channels file.
    """
    return re.sub(r'_[^._]*?\.npz', '_all.pklz', channel_path)


# Maintain a single producer process and multiple consumer processes.
# Use a single blocking queue to coordinate the work from the producer to the consumer
# processes. The producer will keep putting chunks of timesteps to process, blocking
# when the queue gets too full, and the consumers will read from this queue.
#
# Each consumer process will also spawn threads inside of itself to deal with its
# chunked items in parallel. This means we get to use multiple cores on the system via
# Python processes for the consumer processes, as well as being efficient in terms of
# being network/IO-bound using Python threads.


def producer(df, work_queue, year, max_iter):
    # For each timestep group all of its channels together, then group these into chunks that
    # we can efficiently process all at once. Put these into the work queue for consumers
    # to process.
    safeprint('Producer setting up iterable chunks...')
    channels = chunked_iterable(enumerate_timestamp_channels(df, year, max_iter=max_iter),
                                chunk_size=NUM_TIMESTAMPS_AT_ONCE_PER_PROCESS)
    safeprint('Producer finished setting up iterable chunks')
    for chunk in channels:
        work_queue.put(chunk)
        
    for idx in range(NUM_CONSUMERS):
        work_queue.put('DONE')

        
def consumer(process_idx, work_queue, replace):
    """
    Note: this method runs in its own process.
    """
    safeprint('Consumer {} started'.format(process_idx))
    # Note: GCP connections are thread-safe but not multi-process safe.
    client, bucket = connect_gcp(GCP_BUCKET_NAME, GCP_PROJECT_NAME)
    while True:
        msg = work_queue.get()
        if msg == 'DONE':
            safeprint('Consumer {} DONE'.format(process_idx))
            break
        
        chunks = msg
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            executor.map(lambda chunk: process_timestamp(process_idx, chunk, bucket, replace), chunks)

        
def process_timestamp(process_idx, channels, bucket, replace, max_retries=5, wait_time_s=5):
    """
    For each timestamp, download all of its channels in parallel then combine them into a single
    file we can re-upload back to GCP.

    Note: this runs in its own thread, managed by the consumer() method.
    """    
    def download_channel(process_idx, channel_name, channel_path, bucket, max_retries=5, wait_time_s=5):
        """
        Download an individual timestamps channel.
        
        Note: this runs in its own thread, managed by the process_timestamp method().
        """
        for retry in range(max_retries):
            try:
                img_data = bucket.blob(channel_path).download_as_string()
                img = np.load(BytesIO(img_data), allow_pickle=True)['x']
                return (channel_name, img)
            except Exception as e:
                safeprint('Consumer {}: ERROR in download_channel thread for {}: {}, waiting {} seconds for retry {}'
                          .format(process_idx, channel_path, e, wait_time_s, retry))
                time.sleep(wait_time_s)
        
        safeprint('ERROR!!!!!!!! Even after retries unable to fully run download_channel for {}'
                  .format(channel_path))
        
    for retry in range(max_retries):
        try:
            combined_path = get_combined_path(channels[0]['file'])
            if not replace and already_exists(combined_path, bucket):
                safeprint('Consumer {}: File already exists {}'.format(process_idx, combined_path))
                return

            num_workers = len(channels)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(channels)) as executor:
                runme = lambda channel: download_channel(process_idx, channel['channel'], channel['file'], bucket)
                results = executor.map(runme, channels)

                # Ensure we always sort the list of channels consistently by the channel name.
                results = list(results)
                results = sorted(results, key=lambda entry: entry[0])
                
                # Pickle, compress, and upload these to GCP as a single combined channel file.
                compressed_out = bz2.compress(pickle.dumps(results))
                blob = bucket.blob(combined_path)
                blob.upload_from_string(compressed_out)     

                safeprint('Consumer {}: Finished {}'.format(process_idx, combined_path))
                return
        except Exception as e:
            safeprint('Consumer {}: ERROR in process_timestamp thread for {}: {}, waiting {} seconds for retry {}'
                      .format(process_idx, channels, e, wait_time_s, retry))
            time.sleep(wait_time_s)

        safeprint('FINAL ERROR!!!!!!!! Even after retries unable to fully run process_timestamp for {}'
                  .format(channels))


def already_exists(combined_path, bucket):
    """
    Looks to see if a combined file already exists from an earlier run.
    """
    return storage.Blob(bucket=bucket, name=combined_path).exists()
    
    
def main():
    parser = ArgumentParser(description='Pre-process channel files for timestamps into combined files')
    parser.add_argument('--year',
                        dest='year',
                        type=int,
                        required=True,
                        help='Year to process')
    parser.add_argument('--replace',
                        dest='replace',
                        type=bool,
                        default=False,
                        help='If True, will replace pre-existing pre-processed files')

    args = parser.parse_args()
    safeprint('Processing for year {}'.format(args.year))

    safeprint('Connecting to GCP bucket...')
    client, bucket = connect_gcp(GCP_BUCKET_NAME, GCP_PROJECT_NAME)
    safeprint('Connected to GCP bucket')

    safeprint('Processing inventory file...')
    data = bucket.blob(GCP_INVENTORY_PATH).download_as_string()
    df = pd.read_pickle(BytesIO(data), compression='gzip')
    safeprint('Processed inventory file')

    safeprint('Using {} consumers'.format(NUM_CONSUMERS))
    work_queue = mp.Queue(maxsize=NUM_CONSUMERS)
    consumers = [None] * NUM_CONSUMERS
    for idx in range(NUM_CONSUMERS):
        safeprint('Spawning {} consumer...'.format(idx))
        c = mp.Process(target=consumer, args=(idx, work_queue, args.replace))
        c.daemon = False
        c.start()
        safeprint('Spawned {} consumer'.format(idx))

    safeprint('Consumers spawned')

    safeprint('Spawning producer...')
    producer(df, work_queue, args.year, MAX_ITER)

    safeprint('Main process DONE')
    safeprint('Remember to run create_inventory.py to re-create the inventory file')


if __name__ == "__main__":
    main()

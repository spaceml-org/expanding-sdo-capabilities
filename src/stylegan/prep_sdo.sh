#!/bin/bash

sdomldir = ~/SDOML_holdout
outdir = ~/tmp/sdo_for_stylegan
cp dataset_tool_mark.py $outdir/.
python dataset_tool_mark.py create_from_sdo $outdir $sdomldir


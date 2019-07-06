Steps to training StyleGAN on SDOML data

1) git clone https://github.com/NVlabs/stylegan.git
2) Then place the files dataset_tools_mark.py, train_mark.py and script into the newly cloned directory. 
3) mkdir datasets and mkdir datasets/sdo
4) python dataset_tool_mark.py create_from_sdo datasets/sdo $SDOML_basedir. This should generate tfrecords in the datasets/sdo directory.
5) If training on IBM cluster, edit script, then bsub < script. 

#!/bin/sh

bsub -J 'ukb_to_hdf5[1-503]' -o ukb_to_hdf5_out.txt -e ukb_to_hdf5_err.txt '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 00_ukb_to_hdf5.py $LSB_JOBINDEX'

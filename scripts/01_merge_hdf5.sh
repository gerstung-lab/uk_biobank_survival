#!/bin/sh

bsub -J 'merge_job' -n 4 -M 8000 -e merge_err.txt -o merge_out.txt '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 01_merge_hdf5.py' 
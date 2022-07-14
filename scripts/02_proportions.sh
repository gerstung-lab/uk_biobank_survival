#!/bin/sh

bsub -J 'proportions_job' -e proportions_err.txt -o proportions_out.txt '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 02_proportions.py' 
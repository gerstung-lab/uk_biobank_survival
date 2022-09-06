#!/bin/sh

bsub -J 'inference_job' -e inference_err.txt -o inference_out.txt -n 12 -M 12000  '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 03_inference.py' 


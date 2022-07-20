#!/bin/sh

bsub -J 'inference_job7' -e inference_err7.txt -o inference_out7.txt -n 12 -M 12000  '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 03_inference.py' 
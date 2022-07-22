#!/bin/sh

<<<<<<< HEAD
bsub -J 'inference_job7' -e inference_err7.txt -o inference_out7.txt -n 12 -M 12000  '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 03_inference.py' 
=======
bsub -J 'inference_job' -e inference_err.txt -o inference_out.txt -n 12 -M 12000  '/hps/software/users/gerstung/rui/env/conda/envs/penv/bin/python3.9 03_inference.py' 
>>>>>>> add_death_date

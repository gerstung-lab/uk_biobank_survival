
# Modules
#=======================================================================================================================

import sys
import os
import tqdm
import h5py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import Predictive
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import probcox as pcox

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

torch.manual_seed(43)
np.random.seed(7)

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

sys.path.append(ROOT_DIR + 'projects/MultiRisk/scripts/dataloader')
from dataloader import RandomSampler, PIPE, custom_collate, ProportionalSampler

icd10_codes = pd.read_csv(ROOT_DIR + 'projects/MultiRisk/icd10_codes.csv', header=None)
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:]) # removes Source of report of 
icd10_codes = icd10_codes.groupby(0).first()
icd10_code_names = np.squeeze(np.asarray(icd10_codes))
icd10_codes.head()

idx_list = np.load(ROOT_DIR + 'projects/MultiRisk/data/main/idxlist.txt')
   
event_count = np.load(ROOT_DIR + 'projects/MultiRisk/data/main/event_count.txt')

eventlocations = pickle.load(open(ROOT_DIR + 'projects/MultiRisk/data/main/eventlocations.txt', 'rb' ))



#run_id = int(sys.argv[1]) # Variable from cluster
run_id=202
print(run_id)
print(icd10_code_names[run_id])


# Data Pipeline
# =======================================================================================================================

batchsize = 1024
sampling_props = [len(idx_list), batchsize, event_count[run_id], None]
eventlocations[run_id]

pipe = PIPE(file=ROOT_DIR + 'projects/MultiRisk/data/main/ukb.h5', event_idx=run_id)
dataloader = DataLoader(pipe, batch_size=batchsize, num_workers=12, prefetch_factor=1, collate_fn=custom_collate, sampler=ProportionalSampler(idx=idx_list, eventlocations=eventlocations[run_id], iteri=int(10e5), eventsamples=100, randomsamples=batchsize, batchsize=batchsize))

# Inference
#=======================================================================================================================
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_props, predictor=predictor)
m.initialize(eta=0.01, num_particles=5, rank=30) 

loss=[0]
for _, data in tqdm.tqdm(enumerate(dataloader)):
    loss.append(m.infer(data=data))
            
        
g = m.return_guide()
out = g.quantiles([0.025, 0.5, 0.975])

dd = icd10_code_names
dd = np.delete(dd, run_id)
dd[np.argsort(-out['theta'][1].detach().numpy()[:, 0])][:25]

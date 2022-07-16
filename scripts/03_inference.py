
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

torch.manual_seed(4443)
np.random.seed(789)

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

# sys.path.append('/nfs/research/gerstung/rui/uk_biobank_survival/scripts/dataloader')
sys.path.append('./dataloader')
from dataloader import RandomSampler, PIPE, custom_collate, ProportionalSampler

icd10_codes = pd.read_csv('../icd10_codes.csv', header=None)
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:]) # removes Source of report of 
icd10_codes = icd10_codes.groupby(0).first()
icd10_code_names = np.squeeze(np.asarray(icd10_codes))
icd10_codes.head()


idx_list = np.load('../event_files/idxlist.txt.npy')
   
event_count = np.load('../event_files/event_count.txt.npy')

eventlocations = pickle.load(open('../event_files/eventlocations.txt', 'rb' ))


#run_id = int(sys.argv[1]) # Variable from cluster
run_id=114
print(run_id)
print(icd10_code_names[run_id])


# Data Pipeline
# =======================================================================================================================

batchsize = 1024
sampling_props = [len(idx_list), batchsize, event_count[run_id], None]
eventlocations[run_id]

pipe = PIPE(file='../hdf5_files/ukb.h5', event_idx=run_id)
dataloader = DataLoader(pipe, batch_size=batchsize, num_workers=12, prefetch_factor=1, collate_fn=custom_collate, sampler=ProportionalSampler(idx=idx_list, eventlocations=eventlocations[run_id], iteri=int(10e5), eventsamples=100, randomsamples=batchsize, batchsize=batchsize))

# Inference
#=======================================================================================================================
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_props, predictor=predictor,loss=pyro.infer.Trace_ELBO())
m.initialize(eta=0.01, num_particles=5, rank=30,seed=37264) 

loss=[0]


for i, data in tqdm.tqdm(enumerate(dataloader)):
    loss.append(m.infer(data=data))
    if i%100==0:
        np.save('../inference_objects/loss_to_aspergillosis3.npy',loss)
        try:
            m_guide_loc=np.load('../inference_objects/m_to_aspergillosis_guide_loc3.npy')
            xx=np.array(m.guide.loc.cpu().detach().numpy(),ndmin=2)
            m_guide_loc=np.concatenate((m_guide_loc,xx))
            np.save('../inference_objects/m_to_aspergillosis_guide_loc3.npy',m_guide_loc)
            m_guide_scale=np.load('../inference_objects/m_to_aspergillosis_guide_scale3.npy')
            xx=np.array(m.guide.scale.cpu().detach().numpy(),ndmin=2)
            m_guide_scale=np.concatenate((m_guide_scale,xx))
            np.save('../inference_objects/m_to_aspergillosis_guide_scale3.npy',m_guide_scale)
        except:
            m_guide_loc=m.guide.loc.cpu().detach().numpy()
            m_guide_loc=np.array(m_guide_loc,ndmin=2)
            np.save('../inference_objects/m_to_aspergillosis_guide_loc3.npy',m_guide_loc)
            m_guide_scale=m.guide.scale.cpu().detach().numpy()
            m_guide_scale=np.array(m_guide_scale,ndmin=2)
            np.save('../inference_objects/m_to_aspergillosis_guide_scale3.npy',m_guide_scale)
    if i > 200000:
        break
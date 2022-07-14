
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

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

sys.path.append('/nfs/research/gerstung/rui/uk_biobank_survival/scripts/dataloader')
from dataloader import RandomSampler, PIPE, custom_collate

icd10_codes = pd.read_csv('../icd10_codes.csv', header=None)
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:]) # removes Source of report of 
icd10_codes = icd10_codes.groupby(0).first()
icd10_code_names = np.squeeze(np.asarray(icd10_codes))
icd10_codes.head()


f = h5py.File('../hdf5_files/ukb.h5', 'r')
idx_list = list(f.keys())
f.close()    
np.save('../event_files/idxlist.txt', np.asarray(idx_list))

# Data Pipeline
# =======================================================================================================================
class PIPE(Dataset):
    def __init__(self, file):
        """
        """
        self.f = h5py.File(file, 'r')
        
    def __len__(self):
        return(10e100)

    def __getitem__(self, eid):
        time = self.f[eid]['time'][:]
        X = np.zeros(self.f[eid]['dim'][:])
        X[self.f[eid]['row_ind'][:], self.f[eid]['col_ind'][:]] = 1
        X =  np.max(X, axis=0)  
        X = np.concatenate((np.asarray([eid]).astype(float), X), axis=0)
        return(X)
                       
    def __close__(self):
        self.f.close()
        
pipe = PIPE('../hdf5_files/ukb.h5')
dataloader = DataLoader(pipe, batch_size=1, num_workers=1, prefetch_factor=1, collate_fn=None, sampler=RandomSampler(idx=idx_list, iteri=1))

next(iter(RandomSampler(idx=idx_list, iteri=1)))


# Data Loop - iterate once trhough all of UKB - to count occurences of each icd10 code
#=======================================================================================================================
dd=[]
for _, data in tqdm.tqdm(enumerate(dataloader)):
    dd.extend(data.detach().numpy().tolist())
dd = np.asarray(dd)
np.save('../event_files/event_count.txt', np.sum(dd, axis=0)[1:])

dicct = {}
for ii in  tqdm.tqdm(range(1, 1130)):
    dicct[ii-1] = dd[dd[:, ii]==1, 0].tolist()
pickle.dump(dicct, open('../event_files/eventlocations.txt', 'wb' ) )

print(finsihed)

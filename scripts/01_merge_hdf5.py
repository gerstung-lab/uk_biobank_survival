## Modules
#=======================================================================================================================
import os 
import sys 
import glob
import shutil
import tqdm
import h5py

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

with h5py.File(ROOT_DIR + 'projects/MultiRisk/data/ukb.h5', 'w') as h5fw:
    for h5name in tqdm.tqdm(glob.glob(ROOT_DIR + 'projects/MultiRisk/data/tmp/*.h5')):
        helpvar = list(h5fw.keys())
        h5fr = h5py.File(h5name,'r') 
        for obj in h5fr.keys():  
            if obj not in helpvar:
                h5fr.copy(obj, h5fw)  
        h5fr.close()
        

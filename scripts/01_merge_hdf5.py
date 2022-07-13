## Modules
#=======================================================================================================================
import os 
import sys 
import glob
import shutil
import tqdm
import h5py


with h5py.File('../hdf5_files/ukb.h5', 'w') as h5fw:
    for h5name in tqdm.tqdm(glob.glob('../hdf5_files/ukb_*.h5')):
        helpvar = list(h5fw.keys())
        h5fr = h5py.File(h5name,'r') 
        for obj in h5fr.keys():  
            if obj not in helpvar:
                h5fr.copy(obj, h5fw)  
        h5fr.close()
        

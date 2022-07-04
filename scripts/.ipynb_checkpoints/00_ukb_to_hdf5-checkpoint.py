## Modules
#=======================================================================================================================
import os 
import sys 
import shutil
import tqdm
import h5py

import numpy as np 
import pandas as pd 

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

#run_id = int(sys.argv[1]) # Variable from cluster
run_id=1

print(run_id)


## Setup
#=======================================================================================================================
ukb_idx_remove = [1074413, 1220436, 1322418, 1373016, 1484804, 1516618, 1681957, 1898968, 2280037, 2326194, 2492542, 2672990, 2719152, 2753503, 3069616, 3100207, 3114030, 3622841, 3666774, 3683210, 4167470, 4285793, 4335116, 4426913, 4454074, 4463723, 4470984, 4735907, 4739147, 4940729, 5184071, 5938951]

icd10_codes = pd.read_csv(ROOT_DIR + 'projects/MultiRisk/icd10_codes.csv', header=None)
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:]) # removes Source of report of 
icd10_codes = icd10_codes.groupby(0).first()
icd10_code_names = np.squeeze(np.asarray(icd10_codes))
icd10_codes.head()

## Data Prep
#=======================================================================================================================
ukb_iterator = pd.read_csv(ROOT_DIR + 'main/44968/ukb44968.csv', iterator=True, chunksize=1, nrows=1000, skiprows=lambda x: x in np.arange(1, 1000*run_id).tolist()) # iterate over ukb dataset for 1000 per job

for _, dd in tqdm.tqdm(enumerate(ukb_iterator)):
    break
    dd

    # basics - adj., sex, EOO, birthdate
    dd.reset_index(inplace=True)
    dd = dd.astype(str)
    dd = dd.replace('nan', '')

    eid = np.asarray(dd['eid']).astype(int)
    if eid in ukb_idx_remove: # 
        continue
        
    sex = np.asarray(dd['31-0.0']).astype(int)
    birthyear = np.asarray(dd['34-0.0']).astype(str)[0]
    birthmonth = np.asarray(dd['52-0.0']).astype(str)[0]
    if len(birthmonth) == 1:
        birthmonth = '0' + birthmonth
        
    EOO_reason = np.asarray(dd['190-0.0']).astype(str)
    EOO = np.asarray(dd['191-0.0']).astype('datetime64[D]')
    birthdate = np.datetime64(birthyear + '-' + birthmonth, 'D')[None]
    
    if EOO == EOO:
        pass
    else:
        EOO = np.datetime64('2021-01-01', 'D')[None] # set some end date of study

    # extract first occurence data - cat 1712
    d_codes = []
    d_dates = []
    for ii in range(0, 3000, 2):
        try:
            a = np.asarray(dd[[str(130000 + ii) + '-0.0']])[0, 0]
            b = np.asarray(dd[[str(130000 + ii + 1) + '-0.0']])[0, 0]
            if np.logical_and(a != '', b != ''):
                d_codes.extend(np.asarray(icd10_codes.loc[str(130000 + ii + 1) + '-0.0']).tolist())
                d_dates.append(a)
        except:
            pass
    d_dates = np.asarray(d_dates).astype('datetime64[D]')   
    d_codes = np.asarray(d_codes).astype('str')  
    
    # sanity check
    idx = np.logical_and(d_dates>=birthdate, d_dates<=EOO) # should not change anyhting 
    d_dates = d_dates[idx]
    d_codes = d_codes[idx]   
    
    # prepare for intervall format
    dates = np.concatenate((birthdate, d_dates, EOO))
    
    d_codes = np.concatenate((np.asarray(['']),  d_codes, np.asarray(['']))) # expand for birthdate and EOO
    d_codes = (d_codes[:, None] == icd10_code_names).astype(int) # make matrix          
    d_codes.shape                        
    idx_sort = np.argsort(dates)
    dates = dates[idx_sort]
    d_codes = d_codes[idx_sort, :]
    
    # collapse
    d_codes = np.concatenate([np.sum(d_codes[dates==ii, :], axis=0)[None, :] for ii in np.unique(dates)])
    dates = np.unique(dates)

    time_diff = (dates[1:] - dates[:-1]).astype(int)
    time = np.concatenate((np.cumsum(np.concatenate((np.asarray([0]), time_diff)))[:-1, None], np.cumsum(np.concatenate((np.asarray([0]), time_diff)))[1:, None]), axis=1)
    time = np.concatenate((time, np.zeros((time.shape[0], 1))),axis=1)
    
    # extract info of icd10 arrary to save in sparse format
    row_ind, col_ind = np.where(d_codes)
    dim_ = d_codes.shape 
    
    with h5py.File(ROOT_DIR + 'projects/MultiRisk/data/main/ukb_' + str(run_id) + '.h5', 'a') as f:
        f.create_group(str(eid[0]))
        f[str(eid[0])].create_group('MedRec')
        f[str(eid[0])]['MedRec'].create_dataset('time', data=time, maxshape=(None, 3), compression='lzf')
        f[str(eid[0])]['MedRec'].create_dataset('row_ind', data=row_ind, maxshape=(None), compression='lzf')
        f[str(eid[0])]['MedRec'].create_dataset('col_ind', data=col_ind, maxshape=(None), compression='lzf')
        f[str(eid[0])]['MedRec'].create_dataset('dim', data=np.asarray(dim_), maxshape=(2), compression='lzf')

  
# for i in {251..600}; do bsub -env "VAR1=$i" -n 1 -M 1000 -R "rusage[mem=1000]" './00_ukb_to_hdf5.sh'; done



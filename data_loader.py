"""
From Gagan Narula for loading Temple University EEG Corpus.
"""

import numpy as np
import os
from random import shuffle


available_stringlabels = ['null','spsw','gped','pled','eybl','artf','bckg','seiz','fnsz', \
                          'gnsz','spsz','cpsz','absz','tnsz','cnsz','tcsz','atsz','mysz', \
                         'nesz','intr','slow','eyem','chew','shiv','musc','elpp','elst']


def load_processed_data_N_subjects_allchans(subjects_path, Nsub = 1, rand_select = False, matrix_form = True, \
                                   min_samps_if_one_sub = 2000):
    
    subs = os.listdir(subjects_path)
    datafiles = []
    labelfiles = []
    for s in subs:
        if s.endswith('data.npy'):
            datafiles.append(s)
        if s.endswith('label.npy'):
            labelfiles.append(s)
    datafiles = sorted(datafiles)
    labelfiles = sorted(labelfiles)
    if rand_select:
        datafiles, labelfiles = shuffle(datafiles, labelfiles)
    subids = [s.split('/')[-1] for s in datafiles]
    subids = [s.split('_')[0] for s in subids]
    data_per_sub = []
    labels_per_sub = []
    for (d,l) in zip(datafiles[:Nsub], labelfiles[:Nsub]):
        # shape[0] is n_channels, shape[1] is nbatches, shape[2] is nminibatches, 
        # shape[3] is n_steps, shape[4] is ndims 
        data = np.load(os.path.join(subjects_path,d))
        labels = np.load(os.path.join(subjects_path,l)) 
        data = np.squeeze(data)
        labels = np.squeeze(labels)
        if matrix_form:
            if len(data.shape) > 3:
                # this means that there was more than 1 batch 
                # per channel, convert to nsamps x ndims matrix
                data_out = [d.reshape((data.shape[1]*data.shape[2], data.shape[-1])) for d in data]
                data_out = np.array(data_out)
                label_out = labels.reshape((labels.shape[0]*labels.shape[1], labels.shape[-1]))
            else:
                # this means that there was more than 1 batch 
                if data.shape[1] < min_samps_if_one_sub and Nsub==1:
                    print('\n ... Warning: only one subject and has less than %d samples! ...'%(min_samps_if_one_sub))
                data_out = data
                label_out = labels
        data_per_sub.append(data_out)
        labels_per_sub.append(label_out)
    return subids[:Nsub], data_per_sub, labels_per_sub
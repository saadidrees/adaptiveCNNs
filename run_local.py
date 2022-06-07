#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:58:27 2022

@author: saad
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from model.performance import getModelParams
from run_adaptCNNs import run_model


expDate = 'obj_source'
subfold = 'orig_idea'
mdl_name = 'A_CNN_DENSE' #'A_CNN_NORM' #'A_CNN_DENSE' #'CNN_DENSE' 'ACNN_NORM_2DCNN' 'ACNN_NORM_3DCNN' 'ACNN_NORM_2DCNN'
fname_data_train_val_test = ''

info = ''
idxStart_fixedLayers=0
idxEnd_fixedLayers=-1
CONTINUE_TRAINING=0

lr = 0.01           # for CNN_DENSE with few hidden layers, use small LR like 0.001
lr_fac = 1  # how much to divide the learning rate when training is resumed
use_lrscheduler=1
USE_CHUNKER=0
pr_temporal_width = 0
temporal_width=200
thresh_rr=0
chan1_n=10 #200  
filt1_size=1
filt1_3rdDim=0
N_layers = 1
chan2_n=10#100
filt2_size=1
filt2_3rdDim=0
chan3_n=0
filt3_size=0
filt3_3rdDim=0
nb_epochs=100         # setting this to 0 only runs evaluation
bz_ms=1024
BatchNorm=1
MaxPool=0
runOnCluster=0
num_trials=1

BatchNorm_train = 0
saveToCSV=1
trainingSamps_dur=10 # minutes
validationSamps_dur=0
use_batchLogger = 0


path_model_save_base = os.path.join('/home/saad/data/analyses/adaptiveCNNs',expDate,subfold)
path_dataset_base = '/home/saad/data/analyses/adaptiveCNNs/obj_source/datasets/'


c_trial = 1

path_existing_mdl = ''

# %%
for c_trial in range(1,num_trials+1):
    model_performance,mdl = run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,path_dataset_base=path_dataset_base,saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, pr_temporal_width=pr_temporal_width, thresh_rr=thresh_rr,
                            chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                            chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                            chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                            nb_epochs=nb_epochs,bz_ms=bz_ms,N_layers=N_layers,
                            BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,USE_CHUNKER=USE_CHUNKER,
                            path_existing_mdl = path_existing_mdl, idxStart_fixedLayers=idxStart_fixedLayers, idxEnd_fixedLayers=idxEnd_fixedLayers,
                            CONTINUE_TRAINING=CONTINUE_TRAINING,info=info,
                            trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,use_batchLogger=use_batchLogger)
    
plt.plot(model_performance['fev_medianUnits_allEpochs'])
print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))
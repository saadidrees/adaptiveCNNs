#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:44:45 2022

@author: saad
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import lfilter
from scipy.signal import deconvolve
from scipy import integrate
from global_scripts import spiketools

import tensorflow as tf
from tensorflow.keras.layers import Input

import model.models, model.train_model
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer
from model import metrics

import gc
import datetime
import os

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])

from model.data_handler import rolling_window

# %% Build basic signal and response
t = np.arange(0,1,1e-3)
a = 1
b  = 1
stim = (1/(b*t))*(np.exp(a*t))

plt.plot(stim)


# %% Define conditions
range_a = np.arange(0.1,5,.1)
range_b = np.arange(1,1000)

n_test = 100
idx_b_test = np.random.choice(range_b,n_test,replace=False)
range_b_test = range_b[idx_b_test]

# %% create dataset

N_conds = range_a.size * range_b.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_a.size,range_b.size)
conds_mat = np.random.permutation(conds_mat)

stim = np.zeros((N_conds,t.size))
resp = np.zeros((N_conds,1))
conds = np.zeros((N_conds,2))

for i in range(N_conds):
    idx_a,idx_b = np.where(conds_mat==i)
    a = range_a[idx_a]
    b = range_b[idx_b]
    x = (1/b*t)*np.exp(a*t)
    
    stim[i,:] = x
    resp[i,:] = b
    conds[i,0] = b
    conds[i,1] = a
    

plt.plot(stim[:10,:].T)

# %% split the datasets

rgb_idx = np.searchsorted(resp[:,0],range_b_test)
rgb_idx = np.isin(resp[:,0],range_b_test)
stim_val = stim[rgb_idx]
resp_val = resp[rgb_idx]
conds_val = conds[rgb_idx]

stim_train = stim[~rgb_idx]
resp_train = resp[~rgb_idx]
conds_train = conds[rgb_idx]

dict_train = dict(
    X = stim_train,
    y = resp_train,
    conds = conds_train,
    )

dict_val = dict(
    X = stim_val,
    y = resp_val,
    conds = conds_val,
    )


# %% Prepare data
pr_temporal_width = 1000
temporal_width = 900
# % Arrange the data
X = dict_train['X']
X = X[:,:,np.newaxis,np.newaxis]
y = dict_train['y']
if y.ndim==1:
    y = y[:,np.newaxis]
data_train = Exptdata(X,y)

X = dict_val['X']
X = X[:,:,np.newaxis,np.newaxis]
y = dict_val['y']
if y.ndim==1:
    y = y[:,np.newaxis]
data_val = Exptdata(X,y)


# %% Training params

lr = 0.001
nb_epochs = 50
initial_epoch = 0
use_lrscheduler = 1
lr_fac = 1
bz = 1024   
validation_batch_size = bz

c_trial=1

idx_eval = np.arange(0,4000)

# %% 2D CNN

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 900
chan1_n = 1;filt1_size = 1;filt1_3rdDim=0
chan2_n = 0;filt2_size=0;filt2_3rdDim=0
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = False; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl_cnn = model.models.CNN2D_DW(inputs,n_out,**dict_params)
mdl_name_cnn = mdl_cnn.name
mdl_cnn.summary()


fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/func/',mdl_name_cnn,fname_model)
mdl_cnn_history = model.train_model.train(mdl_cnn, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  


# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_cnn.predict(dataset_eval.X[idx_eval])
fev_train_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_cnn.predict(dataset_eval.X[idx_eval])
fev_val_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

plt.plot(mdl_cnn_history.history['fraction_of_explained_variance']);plt.show()
plt.plot(target_eval,pred_eval,'o')
# plt.plot(pred_eval)
plt.show()


# %% Adaptive - conv
inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = temporal_width
chan1_n = 1;filt1_size = 1;filt1_3rdDim=0
chan2_n = 0;filt2_size=1;filt2_3rdDim=1
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = False; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl_adapt = model.models.A_CNN(inputs,n_out,**dict_params)
mdl_adapt.summary()


fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/func/',mdl_adapt.name,fname_model)
mdl_adapt_history = model.train_model.train(mdl_adapt, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

weights_dict = get_weightsDict(mdl_adapt)

# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_adapt.predict(dataset_eval.X[idx_eval])
fev_train_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_adapt.predict(dataset_eval.X[idx_eval])
fev_val_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

plt.plot(mdl_adapt_history.history['fraction_of_explained_variance']);plt.show()
plt.plot(target_eval,pred_eval,'o')
plt.show()

# %%

print('FEV_val_cnn = %0.2f' %fev_val_cnn)
print('FEV_val_adapt = %0.2f' %fev_val_adapt)




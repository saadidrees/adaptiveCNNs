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

import gc
import datetime
import os

from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])



# %% Build basic signal
totalTime = 1 # seconds

f = 1
w = 2*np.pi*f # rad/s
t = np.arange(0,totalTime,1e-3)
stim = np.sin(w*t+np.pi)

plt.plot(stim)

# % Define conditions
range_amps = np.array([40,50,60])
range_freqs = np.arange(1,range_amps.size+1)
range_phases = np.linspace(0,np.pi,range_amps.size)


range_amps_test = np.array([50])
idx_range_amps_test = np.where(range_amps==range_amps_test[0])[0]
range_phases_test = range_phases[idx_range_amps_test]
range_freqs_test = range_freqs[idx_range_amps_test]


range_amps_train = np.setdiff1d(range_amps,range_amps_test)
range_phases_train = np.setdiff1d(range_phases,range_phases_test)
range_freqs_train = np.setdiff1d(range_freqs,range_freqs_test)

# %% TRAIN dataset
num_samps = 100000
num_amps = 2

stim_rgb = np.zeros((stim.size,num_amps,num_samps))
target_train = np.zeros((num_samps,num_amps))

for j in range(num_samps):
    amp_subset_idx =  np.random.choice(range_amps_train.shape[0], num_amps, replace=False)  
    amp_subset = range_amps_train[amp_subset_idx]
    np.random.shuffle(amp_subset)
    
    xs = amp_subset.copy()
    xs.sort()
    rgb_idx = np.searchsorted(range_amps,xs)
    rgb = range_amps[rgb_idx]
    assert np.sum(rgb-xs)==0
    

    for i in range(num_amps):
        amp = amp_subset[i]
        freq = range_freqs_train[amp_subset_idx[i]]
        w = 2*np.pi*freq
        phase = 2*np.pi*np.random.rand(1)

        stim_rgb[:,i,j] = amp*np.sin(w*t+phase)
        
        target_train[j,i] = amp


norm_fac = np.max(range_amps)
stim_rgb = stim_rgb/norm_fac
target_train = target_train/norm_fac

stim_train = np.sum(stim_rgb,axis=1)

plt.plot(stim_rgb[:,:,0])
plt.show()

plt.plot(stim_train[:,:2])
plt.show()
        
assert np.sum(target_train==range_amps_test[0])==0

dict_train = dict(
    X = stim_train,
    y = target_train,
    range_amps_train = range_amps_train,
    range_phases_train = range_phases_train,
    range_freqs_train = range_freqs_train,
    )

# % Test dataset - simply include the ones left out in training
num_samps = 100

stim_rgb = np.zeros((stim.size,num_amps,num_samps))
target_test = np.zeros((num_samps,num_amps))

for j in range(num_samps):
    amp_subset_idx =  np.random.choice(range_amps.shape[0], num_amps, replace=False)  
    if (np.setdiff1d(amp_subset_idx,idx_range_amps_test)).size == num_amps:
        amp_subset_idx = np.concatenate((amp_subset_idx,idx_range_amps_test))
        amp_subset_idx[-num_samps:]
    amp_subset = range_amps[amp_subset_idx]
    

    for i in range(num_amps):
        amp = amp_subset[i]
        freq = range_freqs[amp_subset_idx[i]]
        w = 2*np.pi*freq
    
        phase = 2*np.pi*np.random.rand(1)
        stim_rgb[:,i,j] = amp*np.sin(w*t+phase)
        
        target_test[j,i] = amp

norm_fac = np.max(range_amps)
stim_rgb = stim_rgb/norm_fac
target_test = target_test/norm_fac


stim_test= np.sum(stim_rgb,axis=1)

plt.plot(stim_rgb[:,:,0])
plt.show()

plt.plot(stim_test[:,:2])
plt.show()
        
dict_val = dict(
    X = stim_test,
    y = target_test,
    range_amps_test = range_amps_test,
    range_phases_test = range_phases_test,
    range_freqs_test = range_freqs_test,
    )

# % Arrange the data
X = dict_train['X']
X = np.moveaxis(X,0,-1)
X = X[:,:,np.newaxis,np.newaxis]
y = dict_train['y']
data_train = Exptdata(X,y)

X = dict_val['X']
X = np.moveaxis(X,0,-1)
X = X[:,:,np.newaxis,np.newaxis]
y = dict_val['y']
data_val = Exptdata(X,y)

# %% Create model - Adaptive - conv
lr = 0.0001
c_trial=1
bz = 512   

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 900
chan1_n = 1;filt1_size = 1;filt1_3rdDim=0
chan2_n = 1;filt2_size=1;filt2_3rdDim=1
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = False; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl = model.models.A_CNN(inputs,n_out,**dict_params)
mdl_name = mdl.name

# %% 2D CNN
lr = 0.0001
c_trial=1
bz = 512   

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 2000
chan1_n = 2;filt1_size = 1;filt1_3rdDim=0
chan2_n = 0;filt2_size=0;filt2_3rdDim=0
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = True; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl = model.models.CNN2D(inputs,n_out,**dict_params)
mdl_name = mdl.name



# %% Train model

nb_epochs = 100
validation_batch_size = 100
initial_epoch = 0
use_lrscheduler = 0
lr_fac = 1

fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/sine/',mdl_name,fname_model)
mdl_history = model.train_model.train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

weights_dict = get_weightsDict(mdl)
layer_name = 'conv3d'
weights_layer = get_weightsOfLayer(weights_dict,layer_name)
plt.plot(weights_layer['kernel'])
plt.show()

# %% Evaluate
dataset_eval = data_val
target_eval = dataset_eval.y
pred_eval = mdl.predict(dataset_eval.X)

unit = 0
plt.plot(target_eval[:,unit],pred_eval[:,unit],'o')


# %%

# %% 3D CNNs
X = dict_train['X']
X = np.moveaxis(X,0,-1)
X = X[:,np.newaxis,np.newaxis,np.newaxis,:]
y = dict_train['y']
data_train = Exptdata(X,y)

X = dict_val['X']
X = np.moveaxis(X,0,-1)
X = X[:,np.newaxis,np.newaxis,np.newaxis,:]
y = dict_val['y']
data_val = Exptdata(X,y)

lr = 0.001
c_trial=1
bz = 512   

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 3000
chan1_n = 1;filt1_size = 1;filt1_3rdDim=filt_temporal_width
chan2_n = 0;filt2_size=0;filt2_3rdDim=0
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = True; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,
                   chan1_n=chan1_n,filt1_size=filt1_size,filt1_3rdDim=filt1_3rdDim,
                   chan2_n=chan2_n,filt2_size=filt2_size,filt2_3rdDim=filt2_3rdDim,
                   chan3_n=chan3_n,filt3_size=filt3_size,filt3_3rdDim=filt3_3rdDim,
                   BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl = model.models.CNN3D(inputs,n_out,**dict_params)
mdl_name = mdl.name


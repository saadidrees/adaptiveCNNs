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

from model.data_handler import rolling_window


# %% Build basic signal
totalTime = 1 # mins
timeBin = 10
stim = np.random.rand(int(totalTime*60*1000/timeBin))
stim = np.repeat(stim,timeBin)

plt.plot(stim[100:1000])


# %% Define conditions
range_amps = np.array([20,30,40])
range_spikes = 0.1*range_amps
range_dur = np.array([10,20,50,100])
range_gap = np.array([20,50,100,500,1000])


range_amps_test = np.array([30])
idx_range_amps_test = np.where(range_amps==range_amps_test[0])[0]
range_spikes_test = range_spikes[idx_range_amps_test]
range_dur_test = range_dur
range_gap_test = range_gap

range_amps_train = np.setdiff1d(range_amps,range_amps_test)
range_spikes_train = np.setdiff1d(range_spikes,range_spikes_test)
range_dur_train = range_dur
range_gap_train = range_gap

# %% TRAIN dataset
from scipy.ndimage.filters import gaussian_filter

N_conds = range_amps_train.size * range_dur_train.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_amps_train.size,range_dur_train.size)
conds_mat = np.random.permutation(conds_mat)

stim_train = stim.copy()
spike_vec_train = gaussian_filter(stim_train,sigma=3)

# % embed variations in base signal
idx_prev = 1000

idx_stop = stim_train.shape[0]-(range_dur.max()+(N_conds*range_gap.max()))

while idx_prev<idx_stop:
    for i in range(N_conds):
        idx_amp,idx_dur = np.where(conds_mat==i)
        amp = range_amps[idx_amp]
        dur = range_dur[idx_dur]
        
        rgb = np.random.randint(0,range_dur.size+1)
        idx_next = idx_prev + range_gap[rgb]
        
        idx_sig = np.arange(idx_next,idx_next+dur)
        
        
        stim_train[idx_sig] = stim_train[idx_sig]+amp
        
        spike_vec_train[idx_sig] = spike_vec_train[idx_sig]*range_spikes[idx_amp]
        
        
        idx_prev = idx_next+dur
        idx_prev = idx_prev[0]
    
# normalize
norm_fac = np.max(range_amps)
# stim_train = stim_train/norm_fac

# convert spikes to spikrate
spikeTimes = np.where(spike_vec_train>0)[0]


idx_plots = np.arange(2000,8000)
plt.plot(stim_train[idx_plots])
plt.show()

plt.plot(spike_vec_train[idx_plots])
plt.show()


# plt.plot(spikerate_train[idx_plots])
# plt.show()

# rgb = gaussian_filter(stim_train,sigma=3)
# plt.plot(rgb[idx_plots]/10)
# plt.show()

dict_train = dict(
    X=stim_train,
    y = spike_vec_train,
    range_amps_train = range_amps_train,
    range_spikes_train = range_spikes_train,
    range_dur_train = range_dur_train,
    range_gap_train = range_gap_train,
    )
    
"""
To work on:
    Have two output units.
    One that predicts old intensity and one that predicts the new intensity
"""
# % Test dataset

N_conds = range_amps_test.size * range_dur_test.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_amps_test.size,range_dur_test.size)
conds_mat = np.random.permutation(conds_mat)

stim_test = stim.copy()
stim_test = stim_test[:20000]
spike_vec_test = gaussian_filter(stim_test,sigma=3)

# % embed variations in base signal

idx_prev = 1000

idx_stop = stim_test.shape[0]-(range_dur.max()+(3*range_gap.max()))

while idx_prev<idx_stop:
    for i in range(N_conds):
        idx_amp,idx_dur = np.where(conds_mat==i)
        amp = range_amps[idx_amp]
        dur = range_dur[idx_dur]
        
        rgb = np.random.randint(0,range_dur.size+1)
        idx_next = idx_prev + range_gap[rgb]
        
        idx_sig = np.arange(idx_next,idx_next+dur)
        
        stim_test[idx_sig] = stim_test[idx_sig]+amp
        spike_vec_test[idx_sig] = spike_vec_test[idx_sig]*range_spikes[idx_amp]
        idx_prev = idx_next+dur
        idx_prev = idx_prev[0]
    
# stim_test = stim_test/norm_fac

idx_plots = np.arange(2000,4000)
plt.plot(stim_test[idx_plots])
plt.show()

plt.plot(spike_vec_test[idx_plots])
plt.show()

# plt.plot(spikerate_test)
# plt.show()

dict_val = dict(
    X=stim_test,
    y = spike_vec_test,
    range_amps_test = range_amps_test,
    range_spikes_test = range_spikes_test,
    range_dur_test = range_dur_test,
    range_gap_test = range_gap_test,
    )

# %% Prepare data
pr_temporal_width = 180
temporal_width = 120
# % Arrange the data
X = dict_train['X']
X = rolling_window(X,pr_temporal_width)
X = X[:,:,np.newaxis,np.newaxis]
y = dict_train['y'][pr_temporal_width:]
if y.ndim==1:
    y = y[:,np.newaxis]
data_train = Exptdata(X,y)

X = dict_val['X']
X = rolling_window(X,pr_temporal_width)
X = X[:,:,np.newaxis,np.newaxis]
y = dict_val['y'][pr_temporal_width:]
if y.ndim==1:
    y = y[:,np.newaxis]
data_val = Exptdata(X,y)



# %% 2D CNN
lr = 0.001
c_trial=1
bz = 512   

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 120
chan1_n = 1;filt1_size = 1;filt1_3rdDim=0
chan2_n = 0;filt2_size=0;filt2_3rdDim=0
chan3_n = 0;filt3_size=0;filt3_3rdDim=0
BatchNorm = True; MaxPool = False

dict_params = dict(filt_temporal_width=filt_temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,BatchNorm=BatchNorm,MaxPool=MaxPool)

fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=filt_temporal_width,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                    BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)


mdl = model.models.CNN2D_DW(inputs,n_out,**dict_params)
mdl_name = mdl.name
mdl.summary()


# %% Adaptive - conv
lr = 0.001
c_trial=1
bz = 512   

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = temporal_width
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


mdl = model.models.A_CNN(inputs,n_out,**dict_params)
mdl_name = mdl.name
mdl.summary()

# %% Train model
nb_epochs = 10
validation_batch_size = 500
initial_epoch = 0
use_lrscheduler = 0
lr_fac = 1

fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/lumFluc/',mdl_name,fname_model)
mdl_history = model.train_model.train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

# weights_dict = get_weightsDict(mdl)
# layer_name = 'depthwise_conv2d'
# weights_layer = get_weightsOfLayer(weights_dict,layer_name)
# plt.plot(weights_layer['depthwise_kernel'])
# plt.show()


idx_totake = np.arange(500,2000)
dataset_eval = data_train
target_eval = dataset_eval.y[idx_totake]
pred_eval = mdl.predict(dataset_eval.X[idx_totake])

plt.plot(dataset_eval.X[idx_totake,-1,0,0])
plt.show()

unit = 0
plt.plot(target_eval[:,unit])
plt.plot(pred_eval[:,unit])
plt.show()



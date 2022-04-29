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
from scipy.ndimage.filters import gaussian_filter
from scipy.special import gamma as scipy_gamma

def generate_simple_filter(tau,n,t):
   f = (t**n)*np.exp(-t/tau); # functional form in paper
   f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
   return f

# %% Build basic signal and response
totalTime = 5 # mins
timeBin = 10
stim = np.random.rand(int(totalTime*60*1000/timeBin))
stim = np.repeat(stim,timeBin)
# stim = stim*2

# resp_kern = generate_simple_filter(5,1,np.arange(0,50))
# plt.plot(resp_kern)
# resp = np.convolve(stim,resp_kern)
# resp_fac = 5
# resp = resp*resp_fac

t = np.arange(0,50)
tau = 5
resp_kern = np.exp(-t/tau)
resp_kern = resp_kern/resp_kern.sum()
# resp_kern = np.flip(resp_kern)
resp = np.convolve(stim,resp_kern)

resp_gain = 5
resp = resp_gain*resp

stim = stim[100:]
resp = resp[100:]

idx_plots = np.arange(0,1000)
plt.plot(stim[idx_plots])
plt.plot(resp[idx_plots])

# %% Define conditions
range_amps = np.array([20,30,40,50,60,70,80,90,100])
# range_amps = np.array([1,2])
# range_amps = np.arange(20,100,5)

range_spikes = 1*range_amps
range_dur = np.array([50])
range_gap = np.array([1000])


range_amps_test = np.array([50])
idx_range_amps_test = np.where(np.in1d(range_amps,range_amps_test))[0]
range_spikes_test = range_spikes[idx_range_amps_test]
range_dur_test = range_dur
range_gap_test = range_gap

range_amps_train = np.setdiff1d(range_amps,range_amps_test)
range_spikes_train = np.setdiff1d(range_spikes,range_spikes_test)
# range_amps_train = range_amps
# range_spikes_train = range_spikes
range_dur_train = range_dur
range_gap_train = range_gap

# % TRAIN dataset

N_conds = range_amps_train.size * range_dur_train.size * range_gap_train.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_amps_train.size,range_dur_train.size,range_gap_train.size)
conds_mat = np.random.permutation(conds_mat)

stim_train = stim[:1000000].copy()
spike_vec_train = resp[:1000000].copy() #gaussian_filter(stim_train,sigma=3)

# % embed variations in base signal
idx_prev = 1000

idx_stop = stim_train.shape[0]-(range_dur.max()+(N_conds*range_gap.max()))

for i in range(N_conds):
    idx_amp,idx_dur,idx_gap = np.where(conds_mat==i)
    amp = range_amps_train[idx_amp]
    dur = range_dur_train[idx_dur]
    gap = range_gap_train[idx_gap]
    
    idx_next = idx_prev + gap
    
    idx_sig = np.arange(idx_next,idx_next+dur)
    
    # stim_rgb = stim_train*amp
    # resp_rgb = np.convolve(stim_rgb,resp_kern)
    # stim_train[idx_sig] = stim_rgb[idx_sig]
    # spike_vec_train[idx_sig] = resp_rgb[idx_sig]
    
    stim_train[idx_sig] = stim_train[idx_sig]*amp
    spike_vec_train[idx_sig] = spike_vec_train[idx_sig]+range_spikes_train[idx_amp]
    
    idx_prev = idx_sig[-1]+1
    # idx_prev = idx_prev[0]
idx_prev = idx_prev+500
stim_train = stim_train[:idx_prev]
spike_vec_train = spike_vec_train[:idx_prev]
    
# spike_vec_train = gaussian_filter(spike_vec_train,sigma=2)

# idx_plots = np.arange(2300,3000)
idx_plots = np.arange(0,stim_train.size)
plt.plot(stim_train[idx_plots])
plt.show()


plt.plot(spike_vec_train[idx_plots])
plt.show()


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

N_conds = range_amps_test.size * range_dur_test.size * range_gap_test.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_amps_test.size,range_dur_test.size,range_gap_test.size)
conds_mat = np.random.permutation(conds_mat)

stim_test = stim[:-10000].copy()
spike_vec_test = resp[:-10000].copy()

# % embed variations in base signal

idx_prev = 1000

idx_stop = stim_test.shape[0]-(range_dur.max()+(N_conds+range_gap.max()))
for i in range(N_conds):
    idx_amp,idx_dur,idx_gap = np.where(conds_mat==i)
    amp = range_amps_test[idx_amp]
    dur = range_dur_test[idx_dur]
    gap = range_gap_test[idx_gap]
    
    idx_next = idx_prev + gap
    
    idx_sig = np.arange(idx_next,idx_next+dur)
    
    # stim_rgb = stim_train*amp
    # resp_rgb = np.convolve(stim_rgb,resp_kern)
    # stim_train[idx_sig] = stim_rgb[idx_sig]
    # spike_vec_train[idx_sig] = resp_rgb[idx_sig]

    stim_test[idx_sig] = stim_test[idx_sig]*amp
    spike_vec_test[idx_sig] = spike_vec_test[idx_sig]+range_spikes_test[idx_amp]
    
    idx_prev = idx_sig[-1]
    
idx_prev = idx_prev+500
stim_test = stim_test[:idx_prev]
spike_vec_test = spike_vec_test[:idx_prev]
    
# stim_test = stim_test/norm_fac
# spike_vec_test = gaussian_filter(spike_vec_test,sigma=2)
# idx_plots = np.arange(1500,1550)
idx_plots = np.arange(500,stim_test.size)

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
pr_temporal_width = 600
temporal_width = 500
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


# %% Training params

nb_epochs = 500
initial_epoch = 0
bz = 1024   
validation_batch_size = bz

c_trial=1

idx_eval = np.arange(0,1000)
idx_eval_val = np.arange(0,500)
idx_eval_val_full = np.arange(0,data_val.X.shape[0])


# %%  CNN 2D
lr = 0.01
use_lrscheduler = 0
lr_fac = 1

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = temporal_width
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


mdl_cnn = model.models.CNN2D_DW(inputs,n_out,**dict_params)
mdl_name_cnn = mdl_cnn.name
mdl_cnn.summary()


fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/lumFluc/',mdl_name_cnn,fname_model)
mdl_cnn_history = model.train_model.train(mdl_cnn, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=100,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

weights_cnn = get_weightsDict(mdl_cnn)
# rgb = get_weightsOfLayer(weights_cnn,'depthwise_conv2d')
# plt.plot(rgb['depthwise_kernel'])
# rgb = get_weightsOfLayer(weights_cnn,'dense')
# plt.plot(rgb['kernel'])


# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_cnn.predict(dataset_eval.X[idx_eval])
fev_train_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval = dataset_eval.y[idx_eval_val]
pred_eval = mdl_cnn.predict(dataset_eval.X[idx_eval_val])
fev_val_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval_full = dataset_eval.y[idx_eval_val_full]
pred_eval_full = mdl_cnn.predict(dataset_eval.X[idx_eval_val_full])
fev_val_full_cnn = metrics.fraction_of_explainable_variance_explained(target_eval_full,pred_eval_full,0)


idx_plots = idx_eval_val
plt.plot(mdl_cnn_history.history['fraction_of_explained_variance']);plt.show()
plt.plot(target_eval[idx_plots])
plt.plot(pred_eval[idx_plots])
plt.show()

plt.plot(target_eval_full)
plt.plot(pred_eval_full)
plt.show()


# %% Adaptive - conv
lr = 0.01
use_lrscheduler = 0
lr_fac = 1

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


mdl_adapt = model.models.A_CNN(inputs,n_out,**dict_params)
mdl_adapt.summary()


fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/lumFluc/',mdl_adapt.name,fname_model)
mdl_adapt_history = model.train_model.train(mdl_adapt, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=100,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

weights_dict = get_weightsDict(mdl_adapt)

# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y[idx_eval]
pred_eval = mdl_adapt.predict(dataset_eval.X[idx_eval])
fev_train_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval = dataset_eval.y[idx_eval_val]
pred_eval = mdl_adapt.predict(dataset_eval.X[idx_eval_val])
fev_val_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

dataset_eval = data_val
target_eval_full = dataset_eval.y[idx_eval_val_full]
pred_eval_full = mdl_adapt.predict(dataset_eval.X[idx_eval_val_full])
fev_val_full_adapt = metrics.fraction_of_explainable_variance_explained(target_eval_full,pred_eval_full,0)

idx_plots = idx_eval_val
plt.plot(mdl_adapt_history.history['fraction_of_explained_variance']);plt.show()
plt.plot(target_eval[idx_plots])
plt.plot(pred_eval[idx_plots])
plt.show()

plt.plot(target_eval_full)
plt.plot(pred_eval_full)
plt.show()
# %%

print('FEV_val_cnn = %0.2f' %fev_val_cnn)
print('FEV_val_adapt = %0.2f' %fev_val_adapt)

print('FEV_val_full_cnn = %0.2f' %fev_val_full_cnn)
print('FEV_val_full_adapt = %0.2f' %fev_val_full_adapt)



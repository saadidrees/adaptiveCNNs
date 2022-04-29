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
totalTime = 0.5 # mins
timeBin = 10
stim = np.random.rand(int(totalTime*60*1000/timeBin))
stim = np.repeat(stim,timeBin)


temporal_width = 180
stim = rolling_window(stim,temporal_width)

resp_fac = 1
resp = stim*resp_fac


plt.plot(stim[100:1000,-1])
plt.plot(resp[100:1000,-1])


# %% Define conditions
amp_logFac = 1
range_amps = np.array([2,3,4,5])*amp_logFac
range_dur = np.array([20])
range_gap = np.array([20,30,40,50])


range_amps_test = np.array([4])*amp_logFac
idx_range_amps_test = np.where(np.in1d(range_amps,range_amps_test))[0]
range_dur_test = range_dur
range_gap_test = range_gap

range_amps_train = np.setdiff1d(range_amps,range_amps_test)
range_dur_train = range_dur
range_gap_train = range_gap

# %% TRAIN dataset
from scipy.ndimage.filters import gaussian_filter

N_conds = range_amps_train.size * range_dur_train.size
conds_mat = np.random.permutation(np.arange(0,N_conds,1)).reshape(range_amps_train.size,range_dur_train.size)
conds_mat = np.random.permutation(conds_mat)

stim_train = stim.copy()
spike_vec_train = resp.copy() #gaussian_filter(stim_train,sigma=3)
target_train = np.zeros(stim.shape[0])

# % embed variations in base signal

counter = 0

while counter<stim_train.shape[0]-N_conds+1:
    for i in range(N_conds):
        idx_amp,idx_dur = np.where(conds_mat==i)
        amp = range_amps_train[idx_amp]
        dur = range_dur_train[idx_dur]
        
        rgb = np.random.randint(0,range_dur.size)
        idx_start = range_gap_train[rgb]
        
        idx_sig = np.arange(idx_start,idx_start+dur)
        
        
        stim_train[counter,idx_sig] = stim_train[counter,idx_sig]*amp
        target_train[counter] = amp
        
        counter = counter+1
        
stim_train = stim_train[:counter]
target_train = target_train[:counter]

# normalize
norm_fac = range_amps.max()
stim_train = stim_train/norm_fac
target_train = target_train/norm_fac

idx_plots = 3
plt.plot(stim_train[idx_plots,:])
plt.show()

plt.plot(spike_vec_train[idx_plots])
plt.show()


dict_train = dict(
    X=stim_train,
    y = target_train,
    range_amps_train = range_amps_train,
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
stim_test = stim_test[:5000]
target_test = np.zeros(stim_test.shape[0])
# % embed variations in base signal


counter = 0
while counter<stim_test.shape[0]-N_conds+1:
    for i in range(N_conds):
        idx_amp,idx_dur = np.where(conds_mat==i)
        amp = range_amps_test[idx_amp]
        dur = range_dur_test[idx_dur]
        
        rgb = np.random.randint(0,range_dur_test.size)
        idx_start = range_gap_test[rgb]
        
        idx_sig = np.arange(idx_start,idx_start+dur)
        
        stim_test[counter,idx_sig] = stim_test[counter,idx_sig]*amp
        target_test[counter] = amp
        
        counter = counter + 1

stim_test = stim_test[:counter]
target_test = target_test[:counter]

# normalize
stim_test = stim_test/norm_fac
target_test = target_test/norm_fac


idx_plots = 0
plt.plot(stim_test[idx_plots,:])
plt.show()



dict_val = dict(
    X=stim_test,
    y = target_test,
    range_amps_test = range_amps_test,
    range_dur_test = range_dur_test,
    range_gap_test = range_gap_test,
    )

# %% Prepare data
pr_temporal_width = dict_val['X'].shape[1]
temporal_width = 100
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

lr = 0.0001
nb_epochs = 10
initial_epoch = 0
use_lrscheduler = 0
lr_fac = 1
bz = 512   
validation_batch_size = bz

c_trial=1

idx_eval = np.arange(5000,10000)

# %% 2D CNN

inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer

filt_temporal_width = 120
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
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/lumFluc/',mdl_name_cnn,fname_model)
mdl_history = model.train_model.train(mdl_cnn, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  


# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y
pred_eval = mdl_cnn.predict(dataset_eval.X)
mse_train_cnn = np.mean((pred_eval-target_eval)**2,axis=0)
# fev_train_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

plt.plot(target_eval,pred_eval,'o')
plt.show()


dataset_eval = data_val
target_eval = dataset_eval.y
pred_eval = mdl_cnn.predict(dataset_eval.X)
mse_val_cnn = np.mean((pred_eval-target_eval)**2,axis=0)
# fev_val_cnn = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)

plt.plot(target_eval,pred_eval,'o')
plt.show()


# %% Adaptive - conv
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
mdl_history = model.train_model.train(mdl_adapt, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=1,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  


# Evaluate performance
dataset_eval = data_train
target_eval = dataset_eval.y
pred_eval = mdl_adapt.predict(dataset_eval.X)
mse_train_adapt = np.mean((pred_eval-target_eval)**2,axis=0)
# fev_train_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)
plt.plot(target_eval,pred_eval,'o')
plt.show()

dataset_eval = data_val
target_eval = dataset_eval.y
pred_eval = mdl_adapt.predict(dataset_eval.X)
mse_val_adapt = np.mean((pred_eval-target_eval)**2,axis=0)
# fev_val_adapt = metrics.fraction_of_explainable_variance_explained(target_eval,pred_eval,0)
plt.plot(target_eval,pred_eval,'o')
plt.show()

# plt.hist(target_eval)
# plt.hist(pred_eval)
# plt.show()

# %%

print('MSE_val_cnn = %0.2f' %mse_val_cnn)
print('MSE_val_adapt = %0.2f' %mse_val_adapt)
print('MSE_val_adapt - MSE_val_cnn = %0.2f' %(mse_val_adapt-mse_val_cnn))





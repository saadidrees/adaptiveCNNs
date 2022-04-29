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

from model.data_handler import rolling_window, unroll_data
from scipy.ndimage.filters import gaussian_filter
from scipy.special import gamma as scipy_gamma

def generate_simple_filter(tau,n,t):
   f = (t**n)*np.exp(-t/tau); # functional form in paper
   f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
   return f
 

# %% Build basic signal and response
totalTime = 10 # mins
timeBin = 10
stim = np.random.rand(int(totalTime*60*1000/timeBin))
stim = np.repeat(stim,timeBin)

t = np.arange(0,50)
tau = 2
resp_kern = np.exp(-t/tau)
resp_kern = resp_kern/resp_kern.sum()
plt.plot(resp_kern)

# resp_kern = generate_simple_filter(tau,1,t)


f = np.convolve(stim,resp_kern)
f = f[:stim.shape[0]]
# stim = stim[:-1]

# plt.plot(stim);plt.plot(f); plt.show()

alpha = 0.5
g = alpha*stim

resp = f
# plt.plot(stim);plt.plot(f);plt.plot(g); plt.plot(resp);plt.show()


# resp_fac = 5
# resp = resp+resp_fac

idx_plots = np.arange(0,500)
plt.plot(stim[idx_plots])
plt.plot(resp[idx_plots])
plt.show()

# %% Define conditions
frac_train = 0.95
frac_val = 1-frac_train

idx_train = np.floor(frac_train*stim.shape[0]).astype('int')

stim_train = stim[:idx_train].copy()
spike_vec_train = resp[:idx_train].copy() 

stim_test = stim[idx_train:].copy()
spike_vec_test = resp[idx_train:].copy()

dict_train = dict(
    X=stim_train,
    y = spike_vec_train,
    )


dict_val = dict(
    X=stim_test,
    y = spike_vec_test,
    )

# %% Prepare data
pr_temporal_width = 600
temporal_width = 500
# % Arrange the data
X = dict_train['X']
X = rolling_window(X,pr_temporal_width)
X = X[:,:,np.newaxis,np.newaxis]
y = dict_train['y']
y = rolling_window(y,pr_temporal_width)
y = y[:,y.shape[1]-temporal_width:]
y = y[:,-1]

if y.ndim==1:
    y = y[:,np.newaxis]
data_train = Exptdata(X,y)

X = dict_val['X']
X = rolling_window(X,pr_temporal_width)
X = X[:,:,np.newaxis,np.newaxis]
# y = dict_val['y'][pr_temporal_width:]
y = dict_val['y']
y = rolling_window(y,pr_temporal_width)
y = y[:,y.shape[1]-temporal_width:]
y = y[:,-1]
if y.ndim==1:
    y = y[:,np.newaxis]
data_val = Exptdata(X,y)


# %% Training params

nb_epochs = 100
initial_epoch = 0
bz = 1024   
validation_batch_size = bz

c_trial=1

idx_eval = np.arange(0,1000)
idx_eval_val = np.arange(0,1000)
idx_eval_val_full = np.arange(0,data_val.X.shape[0])


# %% 2D CNN
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


mdl_cnn = model.models.CNN2D_DW(inputs,n_out,**dict_params)
mdl_name_cnn = mdl_cnn.name
mdl_cnn.summary()


fname_excel = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/adaptiveCNNs/temp.csv'
path_model_save = os.path.join('/home/saad/data/analyses/adaptiveCNNs/lumFluc/',mdl_name_cnn,fname_model)
mdl_cnn_history = model.train_model.train(mdl_cnn, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=100,
                    USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  

weights_cnn = get_weightsDict(mdl_cnn)
rgb = get_weightsOfLayer(weights_cnn,'depthwise_conv2d')
plt.plot(rgb['depthwise_kernel'])
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
plt.plot(mdl_cnn_history.history['fraction_of_explained_variance'][2:]);plt.show()
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
BatchNorm = True; MaxPool = False

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
# rgb = get_weightsOfLayer(weights_dict,'dense')
# plt.plot(rgb['kernel']);plt.show()


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
# pred_eval_full = (pred_eval_full-pred_eval_full.min())/(pred_eval_full.max()-pred_eval_full.min())
fev_val_full_adapt = metrics.fraction_of_explainable_variance_explained(target_eval_full,pred_eval_full,0)

idx_plots = idx_eval_val
plt.plot(mdl_adapt_history.history['fraction_of_explained_variance']);plt.show()
plt.plot(target_eval[idx_plots])
plt.plot(pred_eval[idx_plots])
plt.show()

plt.plot(target_eval_full)
plt.plot(pred_eval_full)
plt.show()

print('FEV_val_full_adapt = %0.2f' %fev_val_full_adapt)

# %%

print('FEV_val_cnn = %0.2f' %fev_val_cnn)
print('FEV_val_adapt = %0.2f' %fev_val_adapt)

print('FEV_val_full_cnn = %0.2f' %fev_val_full_cnn)
print('FEV_val_full_adapt = %0.2f' %fev_val_full_adapt)


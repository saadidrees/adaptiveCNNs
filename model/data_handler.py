#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:11:24 2021

@author: saad
"""
 
import numpy as np
import os
import h5py
import math
from scipy.stats import zscore
from collections import namedtuple
from model import utils_si
from model.performance import model_evaluate
import re
   
   
def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        # arr = arr[:,np.newaxis,:,:]
        return array
                 
def unroll_data(data,time_axis=0,rolled_axis=1):
    rgb = data[0]
    rgb = np.concatenate((rgb,data[1:,data.shape[1]-1]),axis=0)
    # rgb = np.concatenate((rgb,data[-1:,0,:]),axis=0)
    return rgb
        


def load_data(fname_dataFile,frac_val=0.2,frac_test=0.05,filt_temporal_width=40,idx_cells=None,thresh_rr=0.45):
    
    # Data
    t_start = 500    # of frames to skip in the begining of stim file (each frame is 16 or 17 ms)   
    
    f = h5py.File(fname_dataFile,'r')
    data_key = 'data'
    stims_keys = list(f[data_key].keys())
    t_frame = np.array(f['data'][stims_keys[0]]['stim_frames'].attrs['t_frame'])
    
    #units
    units_all = np.array(f['/units'])
    units_all = utils_si.h5_tostring(units_all)
    
    if idx_cells is None:
        idx_cells_temp = np.array([np.arange(len(units_all))])
        idx_cells = idx_cells_temp[0,:]
    else:
        idx_cells = idx_cells

    
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    datasets = np.empty(len(stims_keys),dtype='object')
    stim_id = -1*np.ones(len(stims_keys),dtype='int32')
    # total_spikeCounts = np.zeros(len(idx_cells))
    
    # same_stims = ((0,2),(1,3))
    
    for s in range(len(stims_keys)):
        
        # stim info
        code_stim = '/'+data_key+'/'+stims_keys[s]
        stim_len = np.array(f[code_stim+'/spikeRate']).shape[0]
        
        idx_time = np.arange(t_start,stim_len)
         
        # firing rates
        resp = np.array(f[code_stim+'/spikeRate'])[idx_time,idx_cells_temp.T]
        resp = resp.T
        
        # spikeCounts = np.array(f[code_stim+'/spikeCounts'])[idx_time,idx_cells.T]
        # spikeCounts = spikeCounts.T    
       
        #  stim
        stim = np.array(f[code_stim+'/stim_frames'][idx_time,:])
    
        num_CB_x = f[code_stim+'/stim_frames'].attrs['num_checkers_x']   # cb_info['steps_x'][0][0]
        num_CB_y = f[code_stim+'/stim_frames'].attrs['num_checkers_y']   # cb_info['steps_y'][0][0]
        stim_id[s] = f[code_stim+'/stim_frames'].attrs['stim_id']
        
        # # map values to -1 and 1
        # stim = np.round(stim - stim.mean())
        # stim = stim/stim.max()       
        # stim = zscore(stim)
        # Roll for temporal dimension
        stim = np.reshape(stim,(stim.shape[0],num_CB_y,num_CB_x),order='F')       
        stim = rolling_window(stim,filt_temporal_width,time_axis=0) 
        
        resp = resp[filt_temporal_width:,:]
        # spikeCounts = spikeCounts[filt_temporal_width:,:]
        # total_spikeCounts = total_spikeCounts + np.sum(spikeCounts,axis=0)
     
        
        resp_norm = np.empty(resp.shape)
        resp_median = np.empty(resp.shape[1])
        for i in range(0,resp.shape[1]):
            rgb = resp[:,i]
            rgb[rgb==0] = np.nan
            resp_median[i] = np.nanmedian(rgb)
            temp = resp[:,i]/resp_median[i]
            temp[np.isnan(temp)] = 0
            resp_norm[:,i] = temp
        
    
        # idx_train = np.arange(resp_norm.shape[0])
     
        train_stim = stim #[idx_train]
        train_resp_norm = resp_norm #[idx_train,:]
        datasets[s] = Exptdata(train_stim, train_resp_norm)     # array of Exptdata datasets
        
    f.close()
    
    
    del train_stim
    del train_resp_norm
    del resp
    del resp_norm
    del stim
    
    # Check which stimuli files are same i.e. repeats - the same_stims contains index of the datasets that are repeats
    stim_unique = np.unique(stim_id)
    same_stims = np.empty(len(stim_unique),dtype='object')
    
    for i in range(len(stim_unique)):
        a = stim_id == stim_unique[i]
        a = np.where(a)[0]
        same_stims[i] = a
    
    idx_stim_noRepeat = list()
    for i in range(len(same_stims)):
        if same_stims[i].shape[0] < 2:
            idx_stim_noRepeat.append(i)
    
    same_stims = np.delete(same_stims,idx_stim_noRepeat) # the same_stims contains index of the datasets that are repeats
    

# Compute retinal reliability - old [correlation based]
    idx_start = 100
    maxLen = int((60*1000)/t_frame) # just take 60 s #resp.shape[0] - idx_start-1 #15000
    
    dist_cc = np.empty((len(same_stims),len(idx_cells)))
    
    for s in range(len(same_stims)):
        obs_rate = datasets[same_stims[s][0]].y[idx_start:maxLen,:]
        est_rate = datasets[same_stims[s][1]].y[idx_start:maxLen,:]
        x_mu = obs_rate - np.mean(obs_rate, axis=0)
        x_std = np.std(obs_rate, axis=0)
        y_mu = est_rate - np.mean(est_rate, axis=0)
        y_std = np.std(est_rate, axis=0)
        dist_cc[s,:] = np.squeeze(np.mean(x_mu * y_mu,axis=0) / (x_std * y_std))
        
    dist_cc_meanAcrossStims = np.mean(dist_cc,axis=0)       
    
    
# NEW - FEV method
    
    idx_unitsToTake = np.arange(units_all.shape[-1])
    numCells = len(idx_unitsToTake)
    dataset_rr = {}
    # For repeated datasets, put their responses in a single matrix so we can later easily compute variances
    for s in range(len(same_stims)):
        temp = np.empty((len(same_stims[s]),maxLen,numCells))    
        for t in range(len(same_stims[s])):
            temp[t,:,:] = datasets[same_stims[s][t]].y[idx_start:idx_start+maxLen,idx_unitsToTake]
            
            dict_vars = {
                 'subset_train': temp,
                 # 'dataset_name': np.atleast_1d(np.array(dset_name,dtype='bytes'))
                 }
         
            dataset_rr['stim_'+str(s)] = dict_vars

           
    dataset_rr_stims = list(dataset_rr.keys())  # name of unique datasets
    var_rate_uniqueTrials = np.array([]).reshape(0,numCells)    # rows correspond to different repeats; unique stimuli has been concatenated below  
    for s in dataset_rr_stims:
        rate_sameStim_trials = dataset_rr[s]['subset_train']   # [trials,frames,units]
        rate_sameStim_avgTrials = np.nanmean(rate_sameStim_trials,axis=0)
        
        rate_avgTrials_sub = rate_sameStim_trials - rate_sameStim_avgTrials[None,:,:]   # trial mean subtracted from rate [trials,frames,units] --> trial-trial noise in the signal
        var_sameStims = np.mean(rate_avgTrials_sub**2,axis=0)   # variance across trials
        
        var_rate_uniqueTrials = np.concatenate((var_rate_uniqueTrials,var_sameStims),axis=0)    # [(frames from stim_0, frames from stim_1,...), units]
    
    var_noise = np.nanmean(var_rate_uniqueTrials,axis=0)    # estimate of noise
       
    rate_all = np.array([]).reshape(0,numCells) 
    for s in dataset_rr_stims:      # concatenate rate from all trials of all stimuli
        for t in range(dataset_rr[s]['subset_train'].shape[0]):
            rgb = dataset_rr[s]['subset_train'][t,:,:]
            rate_all = np.vstack((rate_all,rgb))
    
    var_rate_all = np.var(rate_all,axis=0)  # total variance in the signal
    fractionExplainableVariance_allUnits = (var_rate_all - var_noise)/var_rate_all
    retinalReliability = np.nanmedian(fractionExplainableVariance_allUnits)

    data_quality = {
        'retinalReliability': retinalReliability,
        'dist_cc': dist_cc_meanAcrossStims,        
        'idx_unitsToTake': idx_unitsToTake,         # old metric for compatibility
        'fev_allUnits': fractionExplainableVariance_allUnits,   # This is for all the units!
        'retinalReliability': retinalReliability,
        'var_noise': var_noise,
        'uname_selectedUnits': units_all,  
        }
    print('Retinal Reliability: '+str(np.round(retinalReliability,2)))
    print('Number of selected cells: ',str(numCells))

    
    # Form validation dataset by taking a chunk out of every unique stimulus file
    stim_unique
    n_total = 0
    for s in range(len(datasets)):
        n_total = n_total + datasets[s].y.shape[0]
    n_val = int(np.ceil(frac_val*n_total))
    
    n_perUniqueStim = int(n_val/len(stim_unique))
    train_resp = np.array([]).reshape(0,numCells)
    train_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
    val_resp = np.array([]).reshape(0,numCells)
    val_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
    
    for s in range(len(datasets)):
        rgb_train_resp = datasets[s].y[:-n_perUniqueStim]
        rgb_train_stim = datasets[s].X[:-n_perUniqueStim]

        if s in stim_unique:
            rgb_val_resp = datasets[s].y[-n_perUniqueStim:]
            rgb_val_stim = datasets[s].X[-n_perUniqueStim:]
            
        else:
            rgb_val_resp = np.array([]).reshape(0,numCells)
            rgb_val_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
 
            
        train_resp = np.concatenate((train_resp,rgb_train_resp),axis=0)
        train_stim = np.concatenate((train_stim,rgb_train_stim),axis=0)
        
        val_resp = np.concatenate((val_resp,rgb_val_resp),axis=0)
        val_stim = np.concatenate((val_stim,rgb_val_stim),axis=0)
            
    data_train = Exptdata(train_stim,train_resp)
    data_val = Exptdata(val_stim,val_resp)
    data_test = data_val;
            
    # make sure no validation or test samples are in training
    if filt_temporal_width > 0:
        a = unroll_data(data_train.X)
        a = a.reshape(a.shape[0],a.shape[1]*a.shape[2])
    else:
        a = data_train.X
    a = np.unique(a,axis=0)
    
    if filt_temporal_width > 0:
        b = unroll_data(data_val.X)
        b = b.reshape(b.shape[0],b.shape[1]*b.shape[2])
        b = b[filt_temporal_width-1:,:]
    else:
        b = data_val.X
    b = np.unique(b,axis=0)
    
    c = np.concatenate((a,b),axis=0)
    train_val_unique = np.unique(c,axis=0)
    train_val_unique = train_val_unique.shape[0]==c.shape[0]
    
    del a, b, c
    
    if train_val_unique is False:
        raise ValueError('Training and Validation sets not unique!')
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr


def load_data_kr_allLightLevels(fname_dataFile,dataset,frac_val=0.2,frac_test=0.05,filt_temporal_width=60,idx_cells_orig=None,thresh_rr=0.15):
    
    valSets = ['scotopic','photopic']   # embed this in h5 file
    
    # Data
    t_start = 0    # of frames to skip in the begining of stim file (each frame is 16 or 17 ms)   
    
    f = h5py.File(fname_dataFile,'r')
    data_key = dataset
    stims_keys = list(f[data_key].keys())
    
    #units
    units_all = np.array(f['/units'])
    units_all = utils_si.h5_tostring(units_all)
    
    if idx_cells_orig is None:
        idx_cells_temp = np.array([np.arange(len(units_all))])
        idx_cells = idx_cells_temp[0,:]
    else:
        idx_cells = idx_cells_orig

    
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    datasets = {}
    resp_non_norm = {}
    # total_spikeCounts = np.zeros(len(idx_cells))
    
    # same_stims = ((0,2),(1,3))
    
    
    for s in stims_keys:
        
        # stim info
        code_stim = '/'+data_key+'/'+s
        stim_len = np.array(f[code_stim+'/spikeRate']).shape[0]        
        idx_time = np.arange(t_start,stim_len)
          
        # firing rates
        resp = np.array(f[code_stim+'/spikeRate'])[idx_time]
        # resp = resp.T
        if resp.ndim == 2:
            resp = resp[:,idx_cells]
            resp = resp[filt_temporal_width:]

        if resp.ndim > 2:
            resp = resp[filt_temporal_width:]
            # resp = np.moveaxis(resp,-1,0)
            resp = resp[:,idx_cells,:]
        
        # spikeCounts = np.array(f[code_stim+'/spikeCounts'])[idx_time,idx_cells.T]
        # spikeCounts = spikeCounts.T    
       
        #  stim
        stim = np.array(f[code_stim+'/stim_frames'][idx_time,:])
        t_frame = f[code_stim+'/stim_frames'].attrs['t_frame']   
        num_CB_x = f[code_stim+'/stim_frames'].attrs['num_checkers_x']   # cb_info['steps_x'][0][0]
        num_CB_y = f[code_stim+'/stim_frames'].attrs['num_checkers_y']   # cb_info['steps_y'][0][0]
        
        stim = np.reshape(stim,(stim.shape[0],num_CB_y,num_CB_x),order='F')       
        # stim = zscore(stim)
        stim = rolling_window(stim,filt_temporal_width,time_axis=0) 
        
        
        # spikeCounts = spikeCounts[filt_temporal_width:,:]
        # total_spikeCounts = total_spikeCounts + np.sum(spikeCounts,axis=0)
    
        
        resp_norm = np.empty(resp.shape)
        
        if s == 'train': #resp_norm.ndim < 3:       Calculate the median based on training data and later apply that to test data
            resp_median = np.empty(resp.shape[1])
            for i in range(0,resp.shape[1]):
                rgb = resp[:,i]
                rgb[rgb==0] = np.nan
                resp_median[i] = np.nanmedian(rgb)
                
                temp = resp[:,i]/resp_median[i]
                temp[np.isnan(temp)] = 0
                resp_norm[:,i] = temp
                resp_orig = resp
            
        else:
            # resp_median = np.empty((resp.shape[1],resp.shape[2]))
            # for j in range(0, resp.shape[2]):
            #     for i in range(0,resp.shape[1]):
            #         rgb = resp[:,i,j]
            #         rgb[rgb==0] = np.nan
            #         resp_median[i,j] = np.nanmedian(rgb)
            temp = resp/resp_median[None,:,None]
            temp[np.isnan(temp)] = 0
            resp_norm = temp
            resp_orig = resp

            
        datasets[s] = Exptdata(stim, resp_norm)
        resp_non_norm[s] = resp_orig
        
    f.close()
    
    
    del resp
    del resp_norm
    del stim
    
    # Check which stimuli are same i.e. repeats
    # stim_unique = np.unique(datasets['train'].X,axis=0)
    
    # select only cells where we wont have so many zero spikerate in a batch   
    if idx_cells_orig is None:
        rgb = np.sum(datasets['train'].y,axis=0)
        thresh = 0 #0.75*np.median(rgb)
        idx_unitsToTake = np.arange(len(units_all))
        idx_unitsToTake = idx_unitsToTake[rgb>thresh]
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median[idx_unitsToTake]
    else:
        idx_unitsToTake = idx_cells
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median
        
# dataset for retinal reliability
    try:
        temp_val = np.moveaxis(datasets['val'].y,-1,0)
        dset_name = np.array(dataset)
     
    except:
        select_val = valSets[-1]
        temp_val = np.moveaxis(datasets['val_'+select_val].y,-1,0)
        dset_name = select_val
    
    dataset_rr = {}    
    if idx_cells_orig is None:
        temp_val = temp_val[:,:,idx_unitsToTake]

    dict_vars = {
         'val': temp_val,
         'dataset_name': np.atleast_1d(np.array(dset_name,dtype='bytes'))
         }
     
    dataset_rr['stim_0'] = dict_vars
     
    
# Retinal reliability method 1 - only take one validation set at this stage


    numCells = len(idx_unitsToTake)
    rate_sameStim_trials = dataset_rr['stim_0']['val']
    _, fracExVar_allUnits, _, corr_allUnits = model_evaluate(rate_sameStim_trials,None,filt_temporal_width,RR_ONLY=True)
    retinalReliability_fev = np.round(np.median(fracExVar_allUnits),2)
    retinalReliability_corr = np.round(np.median(corr_allUnits),2)

    data_quality = {
        'retinalReliability_fev': retinalReliability_fev,
        'retinalReliability_corr': retinalReliability_corr,
        'uname_selectedUnits': units_all,  
        'idx_unitsToTake': idx_unitsToTake,
        'fracExVar_allUnits': fracExVar_allUnits,
        'corr_allUnits': corr_allUnits,
        'resp_median_allUnits' : resp_median_allUnits,
        }
    print('Retinal Reliability - FEV: '+str(np.round(retinalReliability_fev,2)))
    print('Retinal Reliability - Corr: '+str(np.round(retinalReliability_corr,2)))
    print('Number of selected cells: ',str(len(idx_unitsToTake)))
    

    
    # Split dataset into train, validation and test 
    # For validation at model running stage, only take one dataset. 
    try:
        datasets['val'].y.shape
        val_dset_name = 'val'
    except:
        val_dset_name = 'val_'+dset_name
        
    rgb = np.nanmean(datasets[val_dset_name].y,-1)
    if idx_cells_orig is None:
        data_val = Exptdata(datasets[val_dset_name].X,rgb[:,idx_unitsToTake])      # for validation i take the mean rate across all trials
    else:
        data_val = Exptdata(datasets[val_dset_name].X,rgb)
    
    if frac_test>0:
        nsamples_test = int(np.floor(datasets['train'].X.shape[0]*frac_test))       
        idx_test = np.floor(np.arange((datasets['train'].X.shape[0]/2)-(nsamples_test/2),(datasets['train'].X.shape[0]/2)))
        idx_test = np.concatenate((idx_test,np.arange(datasets['train'].X.shape[0]-(nsamples_test/2),datasets['train'].X.shape[0]-1)),axis=0)
        idx_test = idx_test.astype('int')
        if idx_test.shape[0] % 2 != 0:
            idx_test = idx_test[2:]
            # idx_test = np.insert(idx_test,0,idx_test[0]-1)
            

        stim_test = datasets['train'].X[idx_test]
        resp_test = datasets['train'].y[idx_test,:]
        if idx_cells_orig is None:
            resp_test = resp_test[:,idx_unitsToTake]
        data_test = Exptdata(stim_test,resp_test)
        
        idx_train = np.setdiff1d(np.arange(0,datasets['train'].X.shape[0]),idx_test)
        stim_train = datasets['train'].X[idx_train]
        resp_train = datasets['train'].y[idx_train,:]   
        if idx_cells_orig is None:
            resp_train = resp_train[:,idx_unitsToTake] 
        
        data_train = Exptdata(stim_train,resp_train)
        
    else:
        data_test = data_val
        
        if idx_cells_orig is None:
            rgb = datasets['train'].y[:,idx_unitsToTake]
            data_train = Exptdata(datasets['train'].X,rgb)
        else:
            data_train = Exptdata(datasets['train'].X,datasets['train'].y)
    
    resp_orig = {}
    for i in resp_non_norm.keys():
        resp_orig[i] = resp_non_norm[i][:,idx_unitsToTake]
        
    # check_trainVal_contamination(data_train.X,data_val.X)
    # check_trainVal_contamination(data_train.X,data_test.X)
        
                     
    return data_train,data_val,data_test,data_quality,dataset_rr,resp_orig


def prepare_data_cnn3d(data,filt_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,filt_temporal_width,time_axis=0)
    X = np.moveaxis(X,1,-1)
    X = X[:,np.newaxis,:,:,:]
    
    y = data.y[:,idx_unitsToTake]
    y = y[filt_temporal_width:]
    
    data = Exptdata(X,y)
    del X, y
    
    return data

def prepare_data_cnn2d(data,filt_temporal_width,idx_unitsToTake):
    if data != None:
        Exptdata = namedtuple('Exptdata', ['X', 'y'])
        if filt_temporal_width>0:
            X = rolling_window(data.X,filt_temporal_width,time_axis=0)   
            y = data.y[:,idx_unitsToTake]
            y = y[filt_temporal_width:]
        else:
            X = np.expand_dims(data.X,axis=1)
            y = data.y[:,idx_unitsToTake]
        
        data = Exptdata(X,y)
        
        del X, y
    return data

def prepare_data_pr_cnn2d(data,pr_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,pr_temporal_width,time_axis=0)
    y = data.y[:,idx_unitsToTake]
    y = y[pr_temporal_width:]
        
    data = Exptdata(X,y)
    del X, y
    
    return data


def prepare_data_convLSTM(data,filt_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,filt_temporal_width,time_axis=0)
    X = X[:,:,np.newaxis,:,:]
    
    y = data.y[:,idx_unitsToTake]
    y = y[filt_temporal_width:]
    
    data = Exptdata(X,y)
    del X, y
    
    return data

    

def save_h5Dataset(fname,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=None,data_train_info=None,data_val_info=None):
    
    f = h5py.File(fname,'a')
       
    grp = f.create_group('/data_quality')
    keys = list(data_quality.keys())
    for i in range(len(data_quality)):
        if data_quality[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(data_quality[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=data_quality[keys[i]])
            
            
    if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
        data_train_keys = list(data_train.keys())
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train[data_train_keys[0]].X,compression='gzip',chunks=True,maxshape=(None,data_train[data_train_keys[0]].X.shape[-2],data_train[data_train_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_train[data_train_keys[0]].y,compression='gzip',chunks=True,maxshape=(None,data_train[data_train_keys[0]].y.shape[-1]))

        for i in range(1,len(data_train_keys)):
            f['data_train']['X'].resize((f['data_train']['X'].shape[0] + data_train[data_train_keys[i]].X.shape[0]),axis=0)
            f['data_train']['X'][-data_train[data_train_keys[i]].X.shape[0]:] = data_train[data_train_keys[i]].X
            
            f['data_train']['y'].resize((f['data_train']['y'].shape[0] + data_train[data_train_keys[i]].y.shape[0]),axis=0)
            f['data_train']['y'][-data_train[data_train_keys[i]].y.shape[0]:] = data_train[data_train_keys[i]].y


    # if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
    #     for i in data_train.keys():
    #         grp = f.create_group('/'+i)
    #         grp.create_dataset('X',data=data_train[i].X,compression='gzip')
    #         grp.create_dataset('y',data=data_train[i].y,compression='gzip')
            
    else:
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train.X,compression='gzip')
        grp.create_dataset('y',data=data_train.y,compression='gzip')
    
    
    if type(data_val)==dict: # if the training set is divided into multiple datasets
        data_val_keys = list(data_val.keys())
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val[data_val_keys[0]].X,compression='gzip',chunks=True,maxshape=(None,data_val[data_val_keys[0]].X.shape[-2],data_val[data_val_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_val[data_val_keys[0]].y,compression='gzip',chunks=True,maxshape=(None,data_val[data_val_keys[0]].y.shape[-1]))

        for i in range(1,len(data_val_keys)):
            f['data_val']['X'].resize((f['data_val']['X'].shape[0] + data_val[data_val_keys[i]].X.shape[0]),axis=0)
            f['data_val']['X'][-data_val[data_val_keys[i]].X.shape[0]:] = data_val[data_val_keys[i]].X
            
            f['data_val']['y'].resize((f['data_val']['y'].shape[0] + data_val[data_val_keys[i]].y.shape[0]),axis=0)
            f['data_val']['y'][-data_val[data_val_keys[i]].y.shape[0]:] = data_val[data_val_keys[i]].y
    else:
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val.X,compression='gzip')
        grp.create_dataset('y',data=data_val.y,compression='gzip')
    
    if data_test != None:  # data_test is None if it does not exist. So if it doesn't exist, don't save it.
        grp = f.create_group('/data_test')
        grp.create_dataset('X',data=data_test.X,compression='gzip')
        grp.create_dataset('y',data=data_test.y,compression='gzip')
        
    # Training data info
    if type(data_train_info)==dict: # if the training set is divided into multiple datasets
        for i in data_train_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_train_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_train_info[i][j])
    elif data_train_info!=None:
        grp = f.create_group('/data_train_info')
        info_keys = data_train_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_train_info[i][j])
    
    
    # Validation data info
    if type(data_val_info)==dict: # if the training set is divided into multiple datasets
        for i in data_val_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_val_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_val_info[i][j])
    elif data_val_info!=None:
        grp = f.create_group('/data_val_info')
        info_keys = data_val_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_val_info[i][j])

    
    if dataset_rr != None:
        grp = f.create_group('/dataset_rr')
        keys = list(dataset_rr.keys())
        for j in keys:
            grp = f.create_group('/dataset_rr/'+j)
            keys_2 = list(dataset_rr[j].keys())
            for i in range(len(keys_2)):
                if 'bytes' in dataset_rr[j][keys_2[i]].dtype.name:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]])
                elif dataset_rr[j][keys_2[i]].dtype == 'O':
                    grp.create_dataset(keys_2[i], data=np.array(dataset_rr[j][keys_2[i]],dtype='bytes'))
                else:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]],compression='gzip')
                
            
    grp = f.create_group('/parameters')   
    keys = list(parameters.keys())
    for i in range(len(parameters)):
        grp.create_dataset(keys[i], data=parameters[keys[i]]) 
        
    if resp_orig != None:
        grp = f.create_group('/resp_orig')
        keys = list(resp_orig.keys())
        for i in keys:
            grp.create_dataset(i, data=resp_orig[i],compression='gzip') 
            
    f.close()
    
def load_h5Dataset(fname,LOAD_TR=True,LOAD_VAL=True,LOAD_ALL_TR=False,nsamps_val=-1,nsamps_train=-1,RETURN_VALINFO=False):     # LOAD_TR determines whether to load training data or not. In some cases only validation data is required
    
    f = h5py.File(fname,'r')
    t_frame = np.array(f['parameters']['t_frame'])
    # some loading parameters
    if nsamps_val==-1 or nsamps_val==0:
        idx_val_start = 0
        idx_val_end = -1
    else:
        nsamps_val = int((nsamps_val*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_val_start = 1000
        idx_val_end = idx_val_start+nsamps_val
        
        
    if nsamps_train==-1 or nsamps_train==0 :
        idx_train_start = 0
        idx_train_end = -1
    else:
        LOAD_ALL_TR = False
        nsamps_train = int((nsamps_train*60*1000)/t_frame)
        idx_train_start = 1000
        idx_train_end = idx_train_start+nsamps_train

    
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    f_keys = list(f.keys())
    
    # Training data
    if LOAD_TR==True:   # only if it is requested to load the training data
        regex = re.compile(r'data_train_(\d+)')
        dsets = [i for i in f_keys if regex.search(i)]
        if len(dsets)>0:    # if the dataset is split into multiple datasets
            if LOAD_ALL_TR==True:   # concatenate all datasets into one
                X = np.array([]).reshape(0,f[dsets[0]]['X'].shape[1],f[dsets[0]]['X'].shape[2])
                y = np.array([]).reshape(0,f[dsets[0]]['y'].shape[1])
                for i in dsets:
                    rgb = np.array(f[i]['X'])
                    X = np.concatenate((X,rgb),axis=0)
                    
                    rgb = np.array(f[i]['y'])
                    y = np.concatenate((y,rgb),axis=0)
                X = X.astype('float32')
                y = y.astype('float32')
                
                
                                
                data_train = Exptdata(X,y)
            else:            # just pick the first dataset
                X = np.array(f[dsets[0]]['X'][idx_train_start:idx_train_end],dtype='float32')
                y = np.array(f[dsets[0]]['y'][idx_train_start:idx_train_end],dtype='float32')
                data_train = Exptdata(X,y)
            
        else:   # if there is only one dataset
            X = np.array(f['data_train']['X'][idx_train_start:idx_train_end],dtype='float32')
            y = np.array(f['data_train']['y'][idx_train_start:idx_train_end],dtype='float32')
            data_train = Exptdata(X,y)

    else:
        data_train = None
        
    # Validation data
    if LOAD_VAL==True:
        regex = re.compile(r'data_val_(\d+)')
        dsets = [i for i in f_keys if regex.search(i)]
        if len(dsets)>0:
            d = 0
            X = np.array(f[dsets[d]]['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
            y = np.array(f[dsets[d]]['y'][idx_val_start:idx_val_end],dtype='float32')
            data_val = Exptdata(X,y)

                  
            # dataset info
            if ('data_val_info_'+str(d)) in f:
                data_val_info = {}
                for i in f['data_val_info_'+str(d)].keys():
                    data_val_info[i] = np.array(f['data_val_info_'+str(d)][i])
                    
                if 'triggers' in data_val_info:
                    data_val_info['triggers'] = data_val_info['triggers'][idx_val_start:idx_val_end]
                    
            else:
                data_val_info = None
        else:
            X = np.array(f['data_val']['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
            y = np.array(f['data_val']['y'][idx_val_start:idx_val_end],dtype='float32')
            data_val = Exptdata(X,y)

            # dataset info
            if 'data_val_info' in f:
                data_val_info = {}
                for i in f['data_val_info'].keys():
                    data_val_info[i] = np.array(f['data_val_info'][i])
            else:
                data_val_info = None
    else:
        data_val = None

        
        
    
    # Testing data
    if 'data_test' in f.keys():
        data_test = Exptdata(np.array(f['data_test']['X']),np.array(f['data_test']['y']))
    else:       
        data_test = None
    
    # Quality data
    select_groups = ('data_quality')
    level_keys = list(f[select_groups].keys())
    data_quality = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            data_quality[i] = utils_si.h5_tostring(rgb)
        else:
            data_quality[i] = rgb
            
    # Retinal reliability data
    select_groups = ('dataset_rr')
    level_keys = list(f[select_groups].keys())
    dataset_rr = {}
    for i in level_keys:
        level4_keys = list(f[select_groups][i].keys())
        temp_2 = {}

        for d in level4_keys:
            data_key ='/'+select_groups+'/'+i+'/'+d
        
            rgb = np.array(f[data_key])
            try:
                rgb_type = rgb.dtype.name
                if 'bytes' in rgb_type:
                    temp_2[d] = utils_si.h5_tostring(rgb)
                else:
                    temp_2[d] = rgb
            except:
                temp_2[d] = rgb
        dataset_rr[i] = temp_2
        
    # Parameters
    select_groups = ('parameters')
    level_keys = list(f[select_groups].keys())
    parameters = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            parameters[i] = utils_si.h5_tostring(rgb)
        else:
            parameters[i] = rgb
    
    # Orig response (non normalized)
    try:
        select_groups = ('resp_orig')
        level_keys = list(f[select_groups].keys())
        resp_orig = {}
        for i in level_keys:
            data_key = '/'+select_groups+'/'+i
            rgb = np.array(f[data_key])
            rgb_type = rgb.dtype.name
               
            if 'bytes' in rgb_type:
                resp_orig[i] = utils_si.h5_tostring(rgb)
            else:
                resp_orig[i] = rgb
    except:
        resp_orig = None

            
    f.close()

    if RETURN_VALINFO==False:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig,data_val_info
    
    
def check_trainVal_contamination(stimFrames_train,stimFrames_val,filt_temporal_width):
    
    if filt_temporal_width>0:
        rgb = unroll_data(stimFrames_train)
        stimFrames_train_flattened = np.reshape(rgb,(rgb.shape[0],np.prod(rgb.shape[1:])))

        rgb = unroll_data(stimFrames_val)
        stimFrames_val_flattened = np.reshape(rgb,(rgb.shape[0],np.prod(rgb.shape[1:])))
        
    else:
        stimFrames_train_flattened = np.reshape(stimFrames_train,(stimFrames_train.shape[0],np.prod(stimFrames_train.shape[1:])))
        stimFrames_val_flattened = np.reshape(stimFrames_val,(stimFrames_val.shape[0],np.prod(stimFrames_val.shape[1:])))
    
    if np.unique(stimFrames_train_flattened,axis=0).shape[0] != stimFrames_train_flattened.shape[0]:
        Warning('training dataset contains repeated stimulus frames')
    
    if np.unique(stimFrames_val_flattened,axis=0).shape[0] != stimFrames_val_flattened.shape[0]:
        Warning('validation dataset contains repeated stimulus frames')
    
    a = np.unique(stimFrames_train_flattened,axis=0)
    b = np.unique(stimFrames_val_flattened,axis=0)   
    c = np.concatenate((b,a),axis=0)
    d,d_idx = np.unique(c,axis=0,return_index=True)
    # d_idx = d_idx[d_idx>b.shape[0]] - b.shape[0]
    # idx_discard = np.setdiff1d(np.arange(0,a.shape[0]),d_idx)
    idx_discard = np.atleast_1d(np.empty(0))
    if np.abs(np.unique(c,axis=0).shape[0] - c.shape[0]) > 2:
        Warning('Training samples contains validation samples. Finding training indices to remove')
        
        idx_discard = get_index_contamination(stimFrames_train_flattened,stimFrames_val_flattened)
        return idx_discard
    
    else:
        print('No contamination found')
        return idx_discard
              
    # if idx_discard.size!=0:
    #     print('training samples contains validation samples')
    #     Warning('training dataset contains repeated stimulus frames')
    # return idx_discard
    
        
def get_index_contamination(stimFrames_train_flattened,stimFrames_val_flattened):
        
    a = stimFrames_train_flattened
    b = stimFrames_val_flattened
    
    progress_vals = np.arange(0,1,0.1)
    progress = progress_vals*a.shape[0]
    idx_discard = np.atleast_1d([])
    for i in range(a.shape[0]):
        train_frame = a[i]
        for j in range(b.shape[0]):
            val_frame = b[j]
            
            if np.all(train_frame == val_frame):
                idx_discard = np.append(idx_discard,i)
                break
            
        if any(progress == i):
            rgb = progress_vals[progress == i]
            print('progresss: '+str(rgb*100)+'%')
            
    return idx_discard
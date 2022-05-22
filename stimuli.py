#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:01:57 2022

@author: saad
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def obj_source(totalTime,timeBin_obj=10,mean_obj=10,amp_obj=2,timeBin_src = 50,mean_src=10,amp_src=5,sigma=1):
        
        lum_obj = np.random.normal(mean_obj,amp_obj,int(totalTime*60*1000/timeBin_obj))
        lum_obj = np.repeat(lum_obj,timeBin_obj)
        lum_obj = gaussian_filter(lum_obj,sigma=sigma)
        
        numpts = 1e-3*lum_obj.shape[0]
        t = np.arange(0,numpts,1e-3)
        f = 1000/timeBin_src
        w = 2*np.pi*f # rad/s
        lum_src = mean_src + (amp_src*np.sin(w*t+np.pi))
        
        # lum_src = np.random.normal(mean_src,amp_src,int(totalTime*60*1000/timeBin_src))
        # lum_src = np.repeat(lum_src,timeBin_src)
        # lum_src = gaussian_filter(lum_src,sigma=sigma)
        
        
        stim = lum_src*lum_obj
        resp = lum_obj.copy()
        
        return stim,resp,lum_src,lum_obj


def sin_mul(totalTime,freq_obj=5,amp_obj=1,offset_obj=1,freq_src=1,amp_src=1,offset_src=1):
    t = np.arange(0,totalTime,1e-3)

    w = 2*np.pi*freq_obj # rad/s
    lum_obj = offset_obj + (amp_obj*np.sin(w*t+np.pi))
    
    w = 2*np.pi*freq_src # rad/s
    lum_src = offset_src + (amp_src*np.sin(w*t+np.pi))
    

    stim = lum_src*lum_obj
    resp = lum_obj.copy()
    
    return stim,resp,lum_src,lum_obj

    
def obj_source_square(totalTime,timeBin_obj=10,mean_obj=10,amp_obj=2,timeBin_src = 50,mean_src=10,amp_src=5,sigma=1):
    step_block = mean_obj*np.ones(timeBin_obj)
    step_block = np.concatenate((step_block,amp_obj+mean_obj*np.ones(timeBin_obj)),axis=0)
    n_reps = int(np.floor(totalTime*60*1000/step_block.shape[0]))
    step_block = np.tile(step_block,n_reps)
    lum_obj = step_block
    lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    step_block = mean_src*np.ones(timeBin_src)
    step_block = np.concatenate((step_block,amp_src+mean_src*np.ones(timeBin_src)),axis=0)
    n_reps = int(np.ceil(lum_obj.shape[0]/step_block.shape[0]))
    step_block = np.tile(step_block,n_reps)
    lum_src = step_block[:lum_obj.shape[0]]
    lum_src = gaussian_filter(lum_src,sigma=sigma)
    
    stim = lum_src*lum_obj
    resp = lum_obj.copy()
    
    return stim,resp,lum_src,lum_obj




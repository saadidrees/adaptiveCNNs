#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:52:15 2022

@author: saad
"""

from model.parser import parser_run_model


def run_model(expDate,mdl_name,path_model_save_base,fname_data_train_val_test,
                            path_existing_mdl='',idxStart_fixedLayers=0,idxEnd_fixedLayers=-1,
                            saveToCSV=1,runOnCluster=0,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            N_layers=0,temporal_width=40, thresh_rr=0,pr_temporal_width = 0,
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,validationSamps_dur=0,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,
                            lr=0.01,lr_fac=1,use_lrscheduler=1,USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):
 
# %%
    import numpy as np
    import matplotlib.pyplot as plt
    
    from tensorflow.keras.layers import Input
    from tensorflow.keras import Model
    
    import model.models, model.train_model
    from model.performance import get_weightsDict, get_weightsOfLayer
    from model import metrics
    
    import gc
    import datetime
    import os
    import csv
    from scipy.stats import zscore
    from numpy import genfromtxt


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
    totalTime = 10 # mins
    sigma = 2
    
    timeBin_obj = 10
    mean_obj = 10
    amp_obj = 2
    lum_obj = np.random.normal(mean_obj,amp_obj,int(totalTime*60*1000/timeBin_obj))
    lum_obj = np.repeat(lum_obj,timeBin_obj)
    lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    
    timeBin_src = 50
    mean_src = 10
    amp_src = 2
    lum_src = np.random.normal(mean_src,amp_src,int(totalTime*60*1000/timeBin_src))
    lum_src = np.repeat(lum_src,timeBin_src)
    lum_src = gaussian_filter(lum_src,sigma=sigma)
    
    
    stim = lum_src*lum_obj
    resp = lum_obj.copy()
    
    # stim = zscore(stim)
    
    norm_val = np.median(stim)
    stim = stim/norm_val
    # resp = resp/norm_val
        
    # % Datasets
    frac_train = 0.98
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
    
    # % Prepare data
    # % Arrange the data
    X = dict_train['X']
    X = rolling_window(X,temporal_width)
    X = X[:,:,np.newaxis,np.newaxis]
    y = dict_train['y']
    y = rolling_window(y,temporal_width)
    y = y[:,-1]
    
    if y.ndim==1:
        y = y[:,np.newaxis]
    data_train = Exptdata(X,y)
    
    X = dict_val['X']
    X = rolling_window(X,temporal_width)
    X = X[:,:,np.newaxis,np.newaxis]

    y = dict_val['y']
    y = rolling_window(y,temporal_width)
    y = y[:,-1]
    if y.ndim==1:
        y = y[:,np.newaxis]
    data_val = Exptdata(X,y)
    
    inputs = Input(data_train.X.shape[1:]) # keras input layer
    n_out = data_train.y.shape[1]         # number of units in output layer

     
    # %% Build and run model
    
    bz = bz_ms
    validation_batch_size = bz
    initial_epoch = 0
    
   
    dict_params = dict(filt_temporal_width=temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,
                       BatchNorm=BatchNorm,MaxPool=MaxPool,N_layers=N_layers)

    
    fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=temporal_width,
                                                        C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                        C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                        C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                        BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)
    
    
    model_func = getattr(model.models,mdl_name)
    mdl = model_func(inputs, n_out, **dict_params)      
    mdl.summary()
    
    
    fname_excel = 'performance.csv'
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)

    mdl_history = model.train_model.train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=5,
                        USE_CHUNKER=0,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  
    
    
    
    # Evaluate performance
    fev_train = mdl_history.history['fraction_of_explained_variance'][-1]
    fev_val = mdl_history.history['val_fraction_of_explained_variance'][-1]
    print('FEV_train_cnn = %0.2f' %fev_train)
    print('FEV_val_cnn = %0.2f' %fev_val)
    
    plt.plot(mdl_history.history['fraction_of_explained_variance'][50:])

    
# # %% test model outputs
#     fname = '/home/saad/data/analyses/adaptiveCNNs/obj_source/A_CNN_DENSE/U-0.00_P-100_T-100_C1-100-01_C2-200-01_BN-1_MP-0_LR-0.0001_TR-01/weights_U-0.00_P-100_T-100_C1-100-01_C2-200-01_BN-1_MP-0_LR-0.0001_TR-01_epoch-179'
#     mdl.load_weights(fname)
#     weights_dict = get_weightsDict(mdl)
    
#     layer_idx = 15
    
#     x = Input(data_train.X.shape[1:])
#     y = x
#     counter = 0
#     for layer in mdl.layers[1:layer_idx]:
#         if 'operators' in layer.name:
#             y = y[:,:,:,:,-1]
#         else:
#             y = layer(y)
#         counter = counter+1
#     mdl_new = Model(x,y)
#     mdl_new.summary()
    
#     inp = data_train.X[:500]
#     out = mdl_new.predict(inp)
    
#     # plt.plot(inp[:,-1,0,0])

#     if out.ndim>2:
#         plt.plot(out[:,:,0,0])
#     else:
#         plt.plot(out[:,:])


# %% Write performance to csv file
    if np.isnan(fev_train):
        fname_excel_full = os.path.join(path_model_save,fname_excel)
        trainProgress = genfromtxt(fname_excel_full, delimiter=',')
        rgb = np.nanargmax(trainProgress[:,2])
        train_best = trainProgress[rgb,:]
        
        fev_train = train_best[2]
        nb_epochs = train_best[0]


# %%

    if runOnCluster==1:
        path_save_performance = '/home/sidrees/scratch/adaptiveCNNs/performance'
    else:
        path_save_performance = '/home/saad/postdoc_db/projects/adaptiveCNNs/performance'

    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        csv_header = ['mdl_name','expDate','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','N_layers','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_train','FEV_val']
        csv_data = [mdl_name,expDate,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, N_layers, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,BatchNorm,MaxPool,c_trial,fev_train,fev_val]
        
        fname_csv_file = 'performance_'+expDate+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 
        
        
    print('-----FINISHED-----')
    return fev_train, fev_val

        

if __name__ == "__main__":

    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))


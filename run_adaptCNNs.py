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
                            lr=0.01,lr_fac=1,use_lrscheduler=1,use_batchLogger=0,USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',
                            path_dataset_base=''):

# %%
    import numpy as np
    import matplotlib.pyplot as plt
    
    from tensorflow.keras.layers import Input, InputLayer
    from tensorflow.keras import Model
    import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()
    
    import model.models, model.train_model
    from model.load_savedModel import load
    import model.analysis 
    from model import metrics
    
    import gc
    import datetime
    import os
    import csv
    from scipy.stats import zscore
    from numpy import genfromtxt


    from collections import namedtuple
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    from model.data_handler import rolling_window, unroll_data, save_h5Dataset, load_h5Dataset
    from scipy.ndimage.filters import gaussian_filter
    from scipy.special import gamma as scipy_gamma
    import h5py
    
    import stimuli
    
    def generate_simple_filter(tau,n,t):
        t_rep = np.repeat(t[:,None],n.shape[0],axis=-1)
        f = (t_rep**n[None,:])*np.exp(-t_rep/tau[None,:]); # functional form in paper
        f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
        return f
    
    # % Build basic signal and response
    if trainingSamps_dur>0:
        totalTime = trainingSamps_dur
    else:
        totalTime = 10 # mins
        
    if fname_data_train_val_test != '':
        fname = os.path.join(path_dataset_base,fname_data_train_val_test)
        print('loading data from '+fname)

        data_train,data_val,_ = load_h5Dataset(fname)
        
        lum_src = data_train.X[:,-1,0,0]/data_train.y[:,0]
        
        idx_plots = np.arange(200,1200)
        plt.plot(data_train.y[idx_plots],'red');plt.show()
        plt.plot(lum_src[idx_plots]);plt.show()
        plt.plot(data_train.X[idx_plots,-1,0,0],'green');plt.show()



        
    else:
        print('generating training data')
   
        sigma = 1
        
        timeBin_obj = 10
        mean_obj = 5
        amp_obj = 1   
        
        
        mean_src = 5
        timeBin_src = temporal_width #500
        dur_src = np.array([40,60,80])  #np.array([40,60,80]) 
        amp_src = np.array([100,500,1000])#np.array([100,500,1000]) #np.array([10,50,100])
        frac_perturb = 1
    
        
        lum_obj,lum_src = stimuli.obj_source_multi(totalTime=totalTime,timeBin_obj=timeBin_obj,mean_obj=mean_obj,amp_obj=amp_obj,
                                                              timeBin_src = timeBin_src,mean_src=mean_src,amp_src=amp_src,dur_src=dur_src,
                                                              sigma=sigma,temporal_width=temporal_width,frac_perturb=frac_perturb)
    
    
    
        stim = lum_obj+lum_src
        resp = lum_obj
        
        # norm_fac = stim.mean()
        # stim = stim/norm_fac
        # resp = resp/norm_fac
        
        # del lum_obj, lum_src
        
        idx_plots = np.arange(200,1200)
        plt.plot(lum_src[idx_plots])
        plt.plot(lum_obj[idx_plots],'r')
        plt.show()
        plt.plot(stim[idx_plots],'g')
        
        del lum_obj, lum_src
    
            
        # % Datasets
        frac_train = 0.90
        idx_train = np.floor(frac_train*stim.shape[0]).astype('int')
        
        stim_train = stim[:idx_train].copy()
        spike_vec_train = resp[:idx_train].copy() 
        
        stim_test = stim[idx_train:].copy()
        spike_vec_test = resp[idx_train:].copy()
        
        N_test_limit = 5000
        if stim_test.shape[0]>N_test_limit:
            stim_test = stim_test[:N_test_limit]
            spike_vec_test = spike_vec_test[:N_test_limit]
        
        dict_train = dict(
            X=stim_train,
            y = spike_vec_train,
            )
        
        del stim_train, spike_vec_train
        
        dict_val = dict(
            X=stim_test,
            y = spike_vec_test,
            )
        
        del stim_test, spike_vec_test
        
        # % Prepare data
        # % Arrange the data
        # X = dict_train['X']
        # X = np.reshape(X,(int(X.shape[0]/temporal_width),int(temporal_width)),order='C')
        # X = X[:,:,np.newaxis,np.newaxis]
        # y = dict_train['y']
        # y = np.reshape(y,(int(y.shape[0]/temporal_width),int(temporal_width)),order='C')
        # y = y[:,-1] #    y = y[:,-1]
    
        X = dict_train['X']
        X = rolling_window(X,temporal_width)
        X = X[:,:,np.newaxis,np.newaxis]
        y = dict_train['y']
        y = rolling_window(y,temporal_width)
        y = y[:,-1] #    y = y[:,-1]
        
        if y.ndim==1:
            y = y[:,np.newaxis]
        data_train = Exptdata(X,y)
        
        X = dict_val['X']
        X = rolling_window(X,temporal_width)
        X = X[:,:,np.newaxis,np.newaxis]
    
        y = dict_val['y']
        y = rolling_window(y,temporal_width)
        y = y[:,-1] #y = y[:,-1]
        if y.ndim==1:
            y = y[:,np.newaxis]
        data_val = Exptdata(X,y)

        parameters = dict(sigma = sigma,timeBin_obj = timeBin_obj,mean_obj = mean_obj,amp_obj = amp_obj,mean_src = mean_src,timeBin_src = timeBin_src,dur_src = dur_src,amp_src = amp_src,frac_perturb = frac_perturb)
        del X, y

    
    inputs = Input(data_train.X.shape[1:]) # keras input layer
    n_out = data_train.y.shape[1]         # number of units in output layer

    

    
    # %% Build and run model
    
    bz = bz_ms
    validation_batch_size = bz
    initial_epoch = 0
    validation_freq = 5
    
    dropout = 0
   
    dict_params = dict(filt_temporal_width=temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,chan2_n=chan2_n,filt2_size=filt2_size,chan3_n=chan3_n,filt3_size=filt3_size,
                       BatchNorm=BatchNorm,MaxPool=MaxPool,N_layers=N_layers,dropout=dropout)

    
    fname_model,_ = model.models.modelFileName(U=0,P=data_train.X.shape[1],T=temporal_width,
                                                        C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                        N_layers=N_layers,
                                                        C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                        C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                        BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial)
    
    path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    

    if nb_epochs>0:
        if not os.path.exists(path_model_save):
            os.makedirs(path_model_save)


        # if fname_data_train_val_test == '':
        #     fname_savedataset = os.path.join(path_model_save,'stimuli.h5')
        #     save_h5Dataset(fname_savedataset,data_train,data_val,parameters=parameters)
        
    
        model_func = getattr(model.models,mdl_name)
        mdl = model_func(inputs, n_out, **dict_params)      
        
        mdl.summary()
        
        
        fname_excel = 'performance.csv'
    
        mdl_history = model.train_model.train(mdl, data_train, data_val, fname_excel,path_model_save, fname_model, bz=bz, nb_epochs=nb_epochs,validation_batch_size=validation_batch_size,validation_freq=validation_freq,
                            use_batchLogger=use_batchLogger,USE_CHUNKER=USE_CHUNKER,initial_epoch=initial_epoch,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=lr_fac)  
        
        mdl_hist_dict = mdl_history.history
    
    else:
        mdl = load(os.path.join(path_model_save,fname_model))
        fname_history = 'history_'+mdl.name+'.h5'
        fname_history = os.path.join(path_model_save,fname_history)
        f = h5py.File(fname_history,'r')
        mdl_hist_dict = {}
        for key in f.keys():
            mdl_hist_dict[key] = np.array(f[key])
        f.close()
    

    # Evaluate performance
    fev_train = mdl_hist_dict['fraction_of_explained_variance'][-1]
    fev_val = mdl_hist_dict['val_fraction_of_explained_variance'][-1]
    print('FEV_train = %0.2f' %fev_train)
    print('FEV_val = %0.2f' %fev_val)
    
    
    try:
        nb_epochs_temp = len(mdl_hist_dict['fraction_of_explained_variance'])
        t_axis = np.arange(1,nb_epochs_temp+1,1)
        idx_negFEV = np.where(np.array(mdl_hist_dict['fraction_of_explained_variance'])>0)[0][0]
        plt.plot(t_axis[idx_negFEV:],mdl_hist_dict['fraction_of_explained_variance'][idx_negFEV:])
        
        t_axis = np.arange(5,nb_epochs_temp+validation_freq,validation_freq)
        idx_negFEV = np.where(np.array(mdl_hist_dict['val_fraction_of_explained_variance'])>0)[0][0]
        plt.plot(t_axis[idx_negFEV:],mdl_hist_dict['val_fraction_of_explained_variance'][idx_negFEV:])
        
        best_val = np.max(mdl_hist_dict['val_fraction_of_explained_variance'])
        idx_bestVal = np.argmax(mdl_hist_dict['val_fraction_of_explained_variance'])
        idx_bestVal = (idx_bestVal*validation_freq)+validation_freq
        best_val_train = mdl_hist_dict['fraction_of_explained_variance'][idx_bestVal-1]
        
        print('FEV_train = %0.2f' %best_val_train)
        print('FEV_val = %0.2f' %best_val)
        
        # fev_train = best_val_train
        # fev_val = 
    except:
        best_val = 0
        best_val_train = 0

    
    # fname_bestweights = 'weights_'+fname_model+'_epoch-%03d'%idx_bestVal
    # mdl.load_weights(os.path.join(path_model_save,fname_bestweights))
    
    
    # y_pred = mdl.predict(data_val.X)
    # idx_plots = np.arange(0,1500)#data_val.X.shape[0])
    # plt.plot(data_val.X[idx_plots,-1,0,0])
    # plt.plot(5*data_val.y[idx_plots],'darkorange')
    # plt.plot(5*y_pred[idx_plots],'r')
    # plt.show()
    # mdl.evaluate(data_val.X,data_val.y,batch_size=bz)
    
    # y_pred = mdl.predict(data_train.X[:5000])
    # plt.plot(data_train.X[750:5000,-1,0,0])
    # plt.plot(mean_src*data_train.y[750:5000])
    # plt.plot(mean_src*y_pred[750:5000])
    # # plt.plot(data_val.X[:1000,-1,0,0])
    # plt.show()

    # y_real_norm = data_val.y/np.min(data_val.y)
    # y_pred_norm = y_pred/np.min(data_val.y)
    # rgb = model.metrics.fraction_of_explainable_variance_explained(y_real_norm,y_pred_norm,0)

# %% rebuild a-conv output
#     weights_dict = model.analysis.get_weightsDict(mdl)
#     # params_norm = model.analysis.get_weightsOfLayer(weights_dict,'batch_normalization')
#     params = model.analysis.get_weightsOfLayer(weights_dict,'photoreceptor_da_multichan_norm_1')
    
#     # for i in params_updated.keys():
#     #     params[i] = params_updated[i]
    
#     alpha = np.atleast_1d(params['alpha']*params['alpha_mulFac'])
#     beta = np.atleast_1d(params['beta']*params['beta_mulFac'])
#     gamma = np.atleast_1d(params['gamma']*params['gamma_mulFac'])  #np.atleast_1d(0.75) #
#     tauY = np.atleast_1d(params['tauY']*params['tauY_mulFac'])
#     nY = np.atleast_1d(params['nY']*params['nY_mulFac'])
#     tauZ = np.atleast_1d(params['tauZ']*params['tauZ_mulFac']) #np.atleast_1d(50) #
#     nZ =  np.atleast_1d(params['nZ']*params['nZ_mulFac'])       #np.atleast_1d(0.1)
#     tauC = np.atleast_1d(params['tauC']*params['tauC_mulFac'])   #np.atleast_1d(20)
#     nC = np.atleast_1d(params['nC']*params['nC_mulFac']) #np.atleast_1d(0.1) #

    
#     idx_param = 0
#     t = np.arange(0,temporal_width)
#     Ky = generate_simple_filter(tauY,nY,t)
#     Kc = generate_simple_filter(tauC,nC,t)
#     Kz = generate_simple_filter(tauZ,nZ,t)
#     Kz = (gamma*Kc) + (1-gamma)*Kz

    
#     plt.plot(Ky);plt.show()
#     plt.plot(Kz);plt.show()

#     data_toTake = data_val
#     y = np.zeros((data_toTake.X.shape[0],data_toTake.X.shape[1]+Ky.shape[0]-1,Ky.shape[1]))
#     z = np.zeros((data_toTake.X.shape[0],data_toTake.X.shape[1]+Kz.shape[0]-1,Kz.shape[1]))
    
#     # rgb = np.apply_along_axis(lambda m: np.convolve(m, Ky[:,0], mode='full'),axis=1,arr=data_toTake.X[:,:,0,0])
#     # # rgb2 = rgb(data_toTake.X[:,:,0,0])

#     for i in range(Ky.shape[1]):
#         y[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, Ky[:,0], mode='full'),axis=1,arr=data_toTake.X[:,:,0,0]) #np.convolve(stim,Ky[:,i],'full')
#         z[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, Kz[:,0], mode='full'),axis=1,arr=data_toTake.X[:,:,0,0])
#     # y = np.convolve(stim,Ky,'full')
#     # z = np.convolve(stim,Kz,'full')
       
#     y = y[:,:data_toTake.X.shape[1]]
#     z = z[:,:data_toTake.X.shape[1]]
    
#     for i in range(tauC.shape[0]):
#         shift_amt_z = int(tauC[i])
#         shift_amt_y = int(tauY[i])
        
#         z[:,:,i] = z[:,shift_amt_z:shift_amt_z+1,i] #np.roll(z[:,i],-shift_amt)
#         y[:,:,i] = z[:,shift_amt_y:shift_amt_y+1,i] #np.roll(y[:,i],-int(tauY[i]))
    
#     # adapt_out = (alpha[None,:]*y)/(1e-3+(beta[None,:]*z))
    
#     # X = unroll_data(data_toTake.X[:,:,0,0])[temporal_width-1:]
#     # y = data_toTake.y[:,0]
    
#     # plt.plot(stim[:1500]);plt.plot(alpha[idx_param]*y[:1500]);plt.show()
#     # plt.plot(lum_src[:1500]);plt.plot(beta[idx_param]*z[:1500]);plt.show()
#     # plt.plot(lum_obj[100:1100]);plt.plot(adapt_out[100:1100,5:]);plt.show()

#     norm_out = (alpha[None,:]*data_toTake.X[:,:,0,0])/(1+(beta[None,:]*z[:,:,0]))
#     idx_toplot = np.arange(1000,1500)
#     plt.plot(data_toTake.X[idx_toplot,-1,0,0]);plt.plot(norm_out[idx_toplot,-1]);plt.ylim([0,100])
    
#     # # final_out_scaled = (adapt_out - adapt_out.mean()) / (np.var(adapt_out)**.5)
#     # # final_out_scaled = (params_norm['gamma']*final_out_scaled) + params_norm['beta']
#     # # final_out = np.log(1+np.exp(final_out_scaled))
    
#     # final_out = np.log(1+np.exp(adapt_out))
    
    
    
#     # idx_plots = np.arange(1000,2000)
#     # plt.plot(resp[idx_plots])
#     # plt.plot(final_out[idx_plots])
#     # # plt.plot(final_out_scaled[idx_plots])
#     # plt.show()
    
#     # # plt.plot(Ky,'r')
#     # plt.plot(Kz,'m')
#     # plt.show()
    
#     # plt.plot(stim[:500])
#     # # plt.plot(y[:500],'r')
#     # plt.plot(z[:500],'m')
#     # # plt.plot(1/(1+beta*z[:500]))
#     # plt.show()

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
        csv_header = ['mdl_name','expDate','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','N_layers','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_train','FEV_val','FEVBestVal_train','FEVBest_val']
        csv_data = [mdl_name,expDate,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, N_layers, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,BatchNorm,MaxPool,c_trial,fev_train,fev_val,best_val_train,best_val]
        
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
    return mdl_hist_dict['fraction_of_explained_variance'], mdl

        

if __name__ == "__main__":

    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))



# %% [OLD]

    # lum_obj = np.random.normal(mean_obj,amp_obj,int(totalTime*60*1000/timeBin_obj))
    # lum_obj = np.repeat(lum_obj,timeBin_obj)
    # lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    

    # mean_src = 5
    # timeBin_src = 500 #500
    # dur_src = 50
    # amp_src = np.array([10,20,30,40])
    # step_block = mean_src*np.ones((timeBin_src,amp_src.size))
    # step_block[:dur_src,:] = amp_src
    # # step_block = np.concatenate((step_block,amp_src+mean_src*np.ones(timeBin_src)),axis=0)
    # n_reps = int(np.ceil(lum_obj.shape[0]/step_block.shape[0]))
    # step_block_seq = step_block[:,0].copy()
    # for i in range(n_reps):
    #     which_amp = np.random.choice(np.arange(amp_src.size))
    #     step_block_seq = np.concatenate((step_block_seq,step_block[:,which_amp]),axis=0)
    # # step_block_seq = np.tile(step_block,n_reps)
    # lum_src = step_block_seq[:lum_obj.shape[0]]
    # lum_src = gaussian_filter(lum_src,sigma=sigma)

    # stim = lum_src*lum_obj
    # resp = lum_obj.copy()



    # # # % Datasets
    # frac_train = 0.90
    # frac_val = 1-frac_train
    
    # idx_train = np.floor(frac_train*stim.shape[0]).astype('int')
    
    # stim_train = stim[:idx_train].copy()
    # spike_vec_train = resp[:idx_train].copy() 
    
    # stim_test = stim[idx_train:].copy()
    # spike_vec_test = resp[idx_train:].copy()
    
    # N_test_limit = 5000
    # if stim_test.shape[0]>N_test_limit:
    #     stim_test = stim_test[:N_test_limit]
    #     spike_vec_test = spike_vec_test[:N_test_limit]
    
    # dict_train = dict(
    #     X=stim_train,
    #     y = spike_vec_train,
    #     )
    
    
    # dict_val = dict(
    #     X=stim_test,
    #     y = spike_vec_test,
    #     )
    
    # # % Prepare data
    # # % Arrange the data
    # X = dict_train['X']
    # X = rolling_window(X,temporal_width)
    # X = X[:,:,np.newaxis,np.newaxis]
    # y = dict_train['y']
    # y = rolling_window(y,temporal_width)
    # y = y[:,-1] #    y = y[:,-1]
    
    # if y.ndim==1:
    #     y = y[:,np.newaxis]
    # data_train = Exptdata(X,y)
    
    # X = dict_val['X']
    # X = rolling_window(X,temporal_width)
    # X = X[:,:,np.newaxis,np.newaxis]

    # y = dict_val['y']
    # y = rolling_window(y,temporal_width)
    # y = y[:,-1] #y = y[:,-1]
    # if y.ndim==1:
    #     y = y[:,np.newaxis]
    # data_val = Exptdata(X,y)
    
    # inputs = Input(data_train.X.shape[1:]) # keras input layer
    # n_out = data_train.y.shape[1]         # number of units in output layer




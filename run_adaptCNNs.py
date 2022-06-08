#filt1_sizey_predlum_objdata_val.ypred_laya.sh!/usr/bin/env python3
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
    
    from model.data_handler import rolling_window, unroll_data, save_h5Dataset, load_h5Dataset, check_trainVal_contamination
    from scipy.ndimage.filters import gaussian_filter
    from scipy.special import gamma as scipy_gamma
    import h5py
    
    import stimuli
    
    def generate_simple_filter(tau,n,t):
        t_rep = np.repeat(t[:,None],n.shape[0],axis=-1)
        f = (t_rep**n[None,:])*np.exp(-t_rep/tau[None,:]); # functional form in paper
        f = (f/tau**(n+1))/scipy_gamma(n+1) # normalize appropriately
        # f = f/np.sum(f,axis=0)
        return f
    
    # % Build basic signal and response
    if trainingSamps_dur>0:
        totalTime = trainingSamps_dur
    else:
        totalTime = 10 # mins
    
    if fname_data_train_val_test == '0':
        fname_data_train_val_test = ''
        
    if fname_data_train_val_test != '':
        path_dataset_base = os.path.join(path_model_save_base,'datasets')
        fname = os.path.join(path_dataset_base,fname_data_train_val_test)
        print('loading data from '+fname)

        data_train,data_val,parameters = load_h5Dataset(fname)
        mean_src = parameters['mean_src']
        
        lum_src = data_train.X[:,-1,0,0]/data_train.y[:,0]
        
        idx_plots = np.arange(200,10000)
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
        timeBin_src = 500#temporal_width+100
        dur_src = np.array([40,60,80]) 
        amp_src = np.array([1e3]) #[1e1,1e2,1e3,1e4,1e5]) #np.array([10,50,100])
        frac_perturb = 1
    
        
        lum_obj,lum_src = stimuli.obj_source_multi(totalTime=totalTime,timeBin_obj=timeBin_obj,mean_obj=mean_obj,amp_obj=amp_obj,
                                                              timeBin_src = timeBin_src,mean_src=mean_src,amp_src=amp_src,dur_src=dur_src,
                                                              sigma=sigma,temporal_width=temporal_width,frac_perturb=frac_perturb)
    
        # lum_obj,lum_src,_ = stimuli.obj_source_multi_old(totalTime=totalTime,timeBin_obj=timeBin_obj,mean_obj=mean_obj,amp_obj=amp_obj,
        #                                                       timeBin_src = timeBin_src,mean_src=mean_src,amp_src=amp_src,dur_src=dur_src,
        #                                                       sigma=sigma,temporal_width=timeBin_src,frac_perturb=frac_perturb)
        
        

    
        stim = lum_obj*lum_src
        resp = lum_obj
        
        # norm_fac = stim.mean()
        # stim = stim/norm_fac
        # resp = resp/norm_fac
        
        # del lum_obj, lum_src
        
        # For not rolled
        # idx_plots = np.arange(200,205)
        # plt.plot(lum_src[idx_plots,:].T)
        # plt.plot(lum_obj[idx_plots,:].T,'r')
        # plt.show()
        # plt.plot(stim[idx_plots,:].T)
        
        # For rolled
        idx_plots = np.arange(500,4500)
        plt.plot(lum_src[idx_plots])
        plt.plot(lum_obj[idx_plots],'r')
        plt.show()
        plt.plot(stim[idx_plots],'g')

        
        # del lum_obj, lum_src
    
            
        # % Datasets
        frac_train = 0.9
        idx_train = np.floor(frac_train*stim.shape[0]).astype('int')
        
        stim_train = stim[:idx_train].copy()
        spike_vec_train = resp[:idx_train].copy() 
        
        stim_test = stim[idx_train:].copy()
        spike_vec_test = resp[idx_train:].copy()
        
        N_test_limit = 20000
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
        # X = X[:,:,np.newaxis,np.newaxis]
        # y = dict_train['y']
        # y = y[:,temporal_width-1:]
        # # y = y[:,-temporal_width+1]
        # data_train = Exptdata(X,y)
        
        # X = dict_val['X']
        # X = X[:,:,np.newaxis,np.newaxis]
        # y = dict_val['y']
        # y = y[:,temporal_width-1:]
        # # y = y[:,:-temporal_width+1]
        # data_val = Exptdata(X,y)

    
        X = dict_train['X']
        X = rolling_window(X,temporal_width)
        X = X[:,:,np.newaxis,np.newaxis]
        y = dict_train['y']
        y = rolling_window(y,temporal_width)
        y_train_rolled = y.copy()
        y = y[:,-1] #    y = y[:,-1]
        
        if y.ndim==1:
            y = y[:,np.newaxis]
        data_train = Exptdata(X,y)
        
        X = dict_val['X']
        X = rolling_window(X,temporal_width)
        X = X[:,:,np.newaxis,np.newaxis]
    
        y = dict_val['y']
        y = rolling_window(y,temporal_width)
        y_val_rolled = y.copy()
        y = y[:,-1] #y = y[:,-1]
        if y.ndim==1:
            y = y[:,np.newaxis]
        data_val = Exptdata(X,y)
        
        check_trainVal_contamination(y_train_rolled,y_val_rolled,temporal_width)
        

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
    
    if np.isnan(fev_val) or fev_val==float('-inf') or fev_val==float('inf'):
        fev_train = mdl.evaluate(data_train.X,data_train.y,batch_size=5000)
        fev_val = mdl.evaluate(data_val.X,data_val.y,batch_size=data_val.X.shape[0])
    
    print('FEV_train = %0.2f' %fev_train)
    print('FEV_val = %0.2f' %fev_val)
    
    
    best_val = np.max(mdl_hist_dict['val_fraction_of_explained_variance'])
    idx_bestVal = np.argmax(mdl_hist_dict['val_fraction_of_explained_variance'])
    idx_bestVal = (idx_bestVal*validation_freq)+validation_freq
    best_val_train = mdl_hist_dict['fraction_of_explained_variance'][idx_bestVal-1]

    print('FEV_train = %0.2f' %best_val_train)
    print('FEV_val = %0.2f' %best_val)

    try:
        nb_epochs_temp = len(mdl_hist_dict['fraction_of_explained_variance'])
        t_axis = np.arange(1,nb_epochs_temp+1,1)
        idx_negFEV = np.where(np.array(mdl_hist_dict['fraction_of_explained_variance'])>0)[0][0]
        plt.plot(t_axis[idx_negFEV:],mdl_hist_dict['fraction_of_explained_variance'][idx_negFEV:])
        
        t_axis = np.arange(5,nb_epochs_temp+validation_freq,validation_freq)
        idx_negFEV = np.where(np.array(mdl_hist_dict['val_fraction_of_explained_variance'])>0)[0][0]
        plt.plot(t_axis[idx_negFEV:],mdl_hist_dict['val_fraction_of_explained_variance'][idx_negFEV:])
        plt.ylim([-1,1])
        plt.show()
        
    except:
        best_val = 0
        best_val_train = 0

    
    fname_bestweights = 'weights_'+fname_model+'_epoch-%03d'%idx_bestVal
    mdl.load_weights(os.path.join(path_model_save,fname_bestweights))
    
    y_pred = mdl.predict(data_val.X)
    y_act = data_val.y
    X_unr = data_val.X[:,-1,0,0]
    
    idx_plots = np.arange(800,2000)
    plt.plot(X_unr[idx_plots])
    plt.plot(mean_src*y_act[idx_plots],'darkorange')
    plt.plot(mean_src*y_pred[idx_plots],'r')
    plt.ylim([0,50])
    plt.show()
    
    thresh = 100
    idx_src = X_unr>thresh
    
    fev_all = metrics.fraction_of_explainable_variance_explained(y_act,y_pred,obs_noise=0)[0]
    fev_local = metrics.fraction_of_explainable_variance_explained(y_act[~idx_src],y_pred[~idx_src],obs_noise=0)[0]
    fev_global = metrics.fraction_of_explainable_variance_explained(y_act[idx_src],y_pred[idx_src],obs_noise=0)[0]
    
    print('fev_all = %0.2f' %fev_all)
    print('fev_local = %0.2f' %fev_local)
    print('fev_global = %0.2f' %fev_global)

    # y_pred = mdl.predict(data_val.X)
    # idx_plots = np.arange(200,201)#data_val.X.shape[0])
    # plt.plot(data_val.X[idx_plots,temporal_width-1:,0,0].T)
    # plt.plot(mean_src*data_val.y[idx_plots,:].T,'darkorange')
    # plt.plot(mean_src*np.squeeze(y_pred[idx_plots,:].T),'r')
    # plt.ylim([0,50])
    # plt.show()
    # mdl.evaluate(data_val.X,data_val.y,batch_size=bz)
    
    # y_pred = mdl.predict(data_train.X[:5000])
    # plt.plot(data_train.X[750:5000,-1,0,0])
    # plt.plot(mean_src*data_train.y[750:5000])
    # plt.plot(mean_src*y_pred[750:5000])
    # # plt.plot(data_val.X[:1000,-1,0,0])
    # plt.show()

    # y_real_norm = data_val.y
    # y_pred_norm = y_pred
    # rgb = model.metrics.fraction_of_explainable_variance_explained(y_real_norm,y_pred_norm,0)

# %% rebuild a-conv output
    # weights_dict = model.analysis.get_weightsDict(mdl)
    # params_norm = model.analysis.get_weightsOfLayer(weights_dict,'batch_normalization')
    # params = model.analysis.get_weightsOfLayer(weights_dict,'photoreceptor_da_multichan_randinit')
    
    # # for i in params_updated.keys():
    # #     params[i] = params_updated[i]
    
    # zeta = np.atleast_1d(params['zeta']*params['zeta_mulFac'])
    # kappa = np.atleast_1d(params['kappa']*params['kappa_mulFac'])
    # alpha = np.atleast_1d(params['alpha']*params['alpha_mulFac'])
    # beta = np.atleast_1d(params['beta']*params['beta_mulFac'])
    # gamma = np.atleast_1d(params['gamma']*params['gamma_mulFac'])  #np.atleast_1d(0.75) #
    # tauY = np.atleast_1d(params['tauY']*params['tauY_mulFac'])
    # nY = np.atleast_1d(params['nY']*params['nY_mulFac'])
    # tauZ = np.atleast_1d(params['tauZ']*params['tauZ_mulFac']) #np.atleast_1d(50) #
    # nZ =  np.atleast_1d(params['nZ']*params['nZ_mulFac'])       #np.atleast_1d(0.1)
    # tauC = np.atleast_1d(params['tauC']*params['tauC_mulFac'])   #np.atleast_1d(20)
    # nC = np.atleast_1d(params['nC']*params['nC_mulFac']) #np.atleast_1d(0.1) #

    
    # idx_param = 0
    # t = np.arange(0,temporal_width)
    # Ky = generate_simple_filter(tauY,nY,t)
    # Kc = generate_simple_filter(tauC,nC,t)
    # Kz = generate_simple_filter(tauZ,nZ,t)
    # Kz = (gamma*Kc) + (1-gamma)*Kz

    # # Ky = np.flip(Ky)  
    # # Kz = np.flip(Kz)

    # plt.plot(Ky[:50]);plt.show()
    # plt.plot(Kz[:50]);plt.show()

    # data_toTake = data_val

    # y = np.zeros((data_toTake.X.shape[0],data_toTake.X.shape[1]+Ky.shape[0]-1,Ky.shape[1]))
    # z = np.zeros((data_toTake.X.shape[0],data_toTake.X.shape[1]+Kz.shape[0]-1,Kz.shape[1]))
    # for i in range(Ky.shape[1]):
    #     y[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, Ky[:,0], mode='full'),axis=1,arr=data_toTake.X[:,:,0,0]) #np.convolve(stim,Ky[:,i],'full')
    #     z[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, Kz[:,0], mode='full'),axis=1,arr=data_toTake.X[:,:,0,0])
       
    # y = y[:,:-Ky.shape[0]+1]
    # z = z[:,:-Kz.shape[0]+1]
    
    # shift_amt_y = np.argmax(Ky,axis=0).astype('int32')
    # shift_amt_z = np.argmax(Kz,axis=0).astype('int32')
    
    # y_sliced = np.zeros((data_toTake.X.shape[0],Ky.shape[1]))
    # z_sliced = np.zeros((data_toTake.X.shape[0],Kz.shape[1]))
    # for i in range(Ky.shape[1]):
    #     y_sliced[:,i] = y[:,shift_amt_y[i],i] #np.roll(y[:,i],-int(tauY[i]))
    #     z_sliced[:,i] = z[:,shift_amt_z[i],i] #np.roll(z[:,i],-shift_amt)
        
    # y = y_sliced
    # z = z_sliced
    
    # idx_filt = np.arange(0,Ky.shape[1])
    # idx_plots = np.arange(200,1000)
    # adapt_out = zeta[None,:] + (alpha[None,:]*y)/(kappa[None,:]+(beta[None,:]*z))
    # plt.plot(alpha[None,:]*y[idx_plots,:])
    # plt.plot(beta[None,idx_filt]*z[idx_plots,idx_filt])
    # plt.plot(adapt_out[idx_plots,idx_filt])
    # # plt.ylim([0,500])
    
    # # # X = unroll_data(data_toTake.X[:,:,0,0])[temporal_width-1:]
    # # # y = data_toTake.y[:,0]
    
    # # # plt.plot(stim[:1500]);plt.plot(alpha[idx_param]*y[:1500]);plt.show()
    # # # plt.plot(lum_src[:1500]);plt.plot(beta[idx_param]*z[:1500]);plt.show()
    # # # plt.plot(lum_obj[100:1100]);plt.plot(adapt_out[100:1100,5:]);plt.show()

    # # norm_out = (alpha[None,:]*data_toTake.X[:,:,0,0])/(1+(beta[None,:]*z[:,:,0]))
    # # idx_toplot = np.arange(1000,1500)
    # # plt.plot(data_toTake.X[idx_toplot,-1,0,0]);plt.plot(norm_out[idx_toplot,-1]);plt.ylim([0,100])
    
    # # final_out_scaled = (adapt_out - adapt_out.mean()) / (np.var(adapt_out)**.5)
    # # final_out_scaled = (params_norm['gamma']*final_out_scaled) + params_norm['beta']
    # # final_out = np.log(1+np.exp(final_out_scaled))
    
    # # idx_plots = np.arange(1000,2000)
    # # plt.plot(resp[idx_plots])
    # # plt.plot(final_out[idx_plots])
    # # # plt.plot(final_out_scaled[idx_plots])
    # # plt.show()
    
    # # # # # plt.plot(Ky,'r')
    # # # # plt.plot(Kz,'m')
    # # # # plt.show()
    
    # # # # plt.plot(stim[:500])
    # # # # # plt.plot(y[:500],'r')
    # # # # plt.plot(z[:500],'m')
    # # # # # plt.plot(1/(1+beta*z[:500]))
    # # # # plt.show()

# %%
    # yy = inputs
    # for layer in mdl.layers[1:5]:
    #     yy = layer(yy)
    # yy = yy[:,:,:,:,0]
    # # yy = mdl.layers[6](yy)
    # # for layer in mdl.layers[6:9]:
    # #     yy = layer(yy) 
    # new_mdl = Model(inputs,yy)
    # new_mdl.summary()
    
    
    # pred_lay = np.squeeze(new_mdl.predict(data_val.X))
    
    # # step_i = 100
    # idx_plots = np.arange(500,2000)
    # plt.plot(data_val.X[idx_plots,-1,0,0],'black')
    # plt.plot(data_val.y[idx_plots],'b',linewidth=2)
    # plt.plot(pred_lay[idx_plots])
    # plt.ylim([0,50])
    # plt.show()
    
# %% custom dense
    # weights_dict = model.analysis.get_weightsDict(mdl)
    # dense_k = weights_dict['dense/kernel']
    # dense_b = weights_dict['dense/bias']
    
    # uid = 0
    
    # a = pred_lay
    # k = dense_k[None,:]
    # out = a*k
    # out_summed = np.squeeze(np.sum(out,axis=1))
    # out_biased = out_summed + dense_b
    # out_act = np.log(1+np.exp(out_biased))

    # idx_plots = np.arange(900,2000)
    # plt.plot(data_val.X[idx_plots,-1,0,0],'black')
    # plt.plot(mean_src*out_act[idx_plots],'r')
    # plt.ylim([0,50])
    
    # uid = 0
    # plt.plot(out_biased[idx_plots])
    # plt.plot(dense_k)
    # plt.plot(out_act[idx_plots])
    


    
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
        csv_header = ['mdl_name','expDate','temp_window','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','N_layers','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','BatchNorm','MaxPool','c_trial','FEV_train','FEV_val','FEVBestVal_train','FEVBest_val','FEV_all','FEV_local','FEV_global']
        csv_data = [mdl_name,expDate,temporal_width,bz_ms,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, N_layers, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,BatchNorm,MaxPool,c_trial,fev_train,fev_val,best_val_train,best_val,fev_all,fev_local,fev_global]
        
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




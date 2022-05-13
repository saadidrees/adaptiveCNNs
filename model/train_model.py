#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:00:57 2021

@author: saad
"""
import os
import h5py

import tensorflow as tf
import tensorflow.keras.callbacks as cb
from tensorflow.keras.optimizers import Adam, SGD
import model.metrics as metrics
from model import customCallbacks
import numpy as np

def chunker(data,batch_size,mode='default'):
    if mode=='predict': # in predict mode no need to do y
        X = data
        counter = 0
        while True:
            counter = (counter + 1) % X.shape[0]
            for cbatch in range(0, X.shape[0], batch_size):
                yield (X[cbatch:(cbatch + batch_size)])
    else:
        X = data.X
        y = data.y
            
        counter = 0
        while True:
            counter = (counter + 1) % X.shape[0]
            for cbatch in range(0, X.shape[0], batch_size):
                yield (X[cbatch:(cbatch + batch_size)], y[cbatch:(cbatch + batch_size)])
     
        
def lr_scheduler(epoch,lr):
    # [epoch,lr divide factor]
    arr_scheduler = np.array([[15,10],
                              [200,10]])
    
    idx = np.where(arr_scheduler[:,0]==epoch)[0]
    
    if idx.size>0:
        idx = idx[0]
        lr_fac = arr_scheduler[idx,1]
        lr = lr/lr_fac
    
    return lr

# %%
def train(mdl, data_train, data_val,fname_excel,path_model_save, fname_model, bz=588, nb_epochs=200, validation_batch_size=5000,validation_freq=10,USE_CHUNKER=0,initial_epoch=1,lr=0.001,lr_fac=1,use_lrscheduler=0,use_batchLogger=0):
    
    optimizer = Adam(lr) #Adam(lr,clipnorm=1,clipvalue=1) # clipvalue=0.3 # clipnorm=1,
    mdl.compile(loss='poisson', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)
    # mdl.compile(loss='mse', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)


    if initial_epoch==0: # 
        mdl.save(os.path.join(path_model_save,fname_model)) # save model architecture

    if initial_epoch>0:
        try:
            weight_file = 'weights_'+fname_model+'_epoch-%03d' % initial_epoch
            mdl.load_weights(os.path.join(path_model_save,weight_file))

        except:
            weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % initial_epoch
            mdl.load_weights(os.path.join(path_model_save,weight_file))
            

        tf.keras.backend.set_value(mdl.optimizer.learning_rate, lr/lr_fac)  # lr_fac controls how much to divide the learning rate whenever training is resumed

            
    # define model callbacks
    fname_cb = 'weights_'+ fname_model + '_epoch-{epoch:03d}' 
    fname_batch = fname_model + '_batchLog'
    fname_batch = os.path.join(path_model_save,fname_batch)
    
    cbs = [cb.ModelCheckpoint(os.path.join(path_model_save, fname_cb),monitor='fraction_of_explained_variance',save_weights_only=True),  # ,save_best_only=True,mode='auto'
           tf.compat.v1.keras.callbacks.TensorBoard(log_dir=path_model_save, histogram_freq=1, write_grads=True),
           cb.CSVLogger(os.path.join(path_model_save, fname_excel)),
           customCallbacks.displayLog(), #]
           customCallbacks.TerminateOnNaN()]
   
    if use_batchLogger==1:
        for layer in mdl.layers:
            mdl.add_metric(layer.output,name=layer.name)

        cbs.append(customCallbacks.NBatchCSVLogger(fname_batch,path_model_save,separator=',', append=False))
    
    if use_lrscheduler==1:
        cbs.append(cb.LearningRateScheduler(lr_scheduler))

    if USE_CHUNKER==0:  # load all data into gpu ram
        mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
                              callbacks=cbs, validation_data=(data_val.X,data_val.y), validation_freq=validation_freq, shuffle=False, initial_epoch=initial_epoch,use_multiprocessing=True)    # validation_batch_size=validation_batch_size,  validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)
        
    else:
        batch_size = bz
        steps_per_epoch = int(np.ceil(data_train.X.shape[0]/batch_size))
        gen_train = chunker(data_train,batch_size)
        mdl_history = mdl.fit(gen_train,steps_per_epoch=steps_per_epoch,epochs=nb_epochs,callbacks=cbs, validation_data=(data_val.X,data_val.y),shuffle=True,initial_epoch=initial_epoch,use_multiprocessing=True,validation_freq=validation_freq)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val) # steps_per_epoch=steps_per_epoch

      
    rgb = mdl_history.history
    keys = list(rgb.keys())
    
    fname_history = 'history_'+mdl.name+'.h5'
    fname_history = os.path.join(path_model_save,fname_history)                            
    f = h5py.File(fname_history,'w')
    for i in range(len(rgb)):
        f.create_dataset(keys[i], data=rgb[keys[i]])
    f.close()
    
        
    return mdl_history


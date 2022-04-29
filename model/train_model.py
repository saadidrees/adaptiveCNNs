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
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
import model.metrics as metrics
import numpy as np
import re
from tensorflow import keras
from model.load_savedModel import load

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
    
class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print("LR - {}".format(self.model.optimizer.learning_rate))
        

def lr_scheduler(epoch,lr):
    if epoch == 40:
        lr = lr/10
        
    # elif epoch == 100:
    #     lr = lr/10
    return lr

# %%
def train(mdl, data_train, data_val,fname_excel,path_model_base, fname_model, bz=588, nb_epochs=200, validation_batch_size=5000,validation_freq=10,USE_CHUNKER=0,initial_epoch=1,lr=0.001,lr_fac=1,use_lrscheduler=0):
    
    optimizer = Adam(lr,clipnorm=1) # clipvalue=0.3
    mdl.compile(loss='mse', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)
    # mdl.compile(loss='poisson', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)

    if initial_epoch==0: # 
        mdl.save(os.path.join(path_model_base,fname_model)) # save model architecture

    if initial_epoch>0:
        try:
            weight_file = 'weights_'+fname_model+'_epoch-%03d' % initial_epoch
            mdl.load_weights(os.path.join(path_model_base,weight_file))

        except:
            weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % initial_epoch
            mdl.load_weights(os.path.join(path_model_base,weight_file))
            

        tf.keras.backend.set_value(mdl.optimizer.learning_rate, lr/lr_fac)  # lr_fac controls how much to divide the learning rate whenever training is resumed

            
    # mdl.compile(loss='mean_squared_error', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev],experimental_run_tf_function=False)


    # define model callbacks
    fname_cb = 'weights_'+ fname_model + '_epoch-{epoch:03d}' 
    
    cbs = [cb.ModelCheckpoint(os.path.join(path_model_base, fname_cb),monitor='val_loss',save_weights_only=True),  # ,save_best_only=True,mode='auto'
           cb.TensorBoard(log_dir=path_model_base, histogram_freq=0, write_grads=False),
           CustomCallback()]
            # cb.ReduceLROnPlateau(monitor='loss',min_lr=1e-6, factor=0.2, patience=5),
   
    if use_lrscheduler==1:
        cbs.append(cb.LearningRateScheduler(lr_scheduler))

    if USE_CHUNKER==0:  # load all data into gpu ram
        mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=bz, epochs=nb_epochs,
                              callbacks=cbs, validation_data=(data_val.X,data_val.y), validation_freq=validation_freq, shuffle=True, initial_epoch=initial_epoch,use_multiprocessing=True)    # validation_batch_size=validation_batch_size,  validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val)
        
    else:
        batch_size = bz
        steps_per_epoch = int(np.ceil(data_train.X.shape[0]/batch_size))
        gen_train = chunker(data_train,batch_size)
        # gen_val = chunker(data_val,validation_batch_size)
        # mdl_history = mdl.fit(gen_train,steps_per_epoch=steps_per_epoch,epochs=nb_epochs,callbacks=cbs, validation_data=gen_val,shuffle=True,initial_epoch=initial_epoch,use_multiprocessing=True,validation_freq=validation_freq)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val) # steps_per_epoch=steps_per_epoch
        mdl_history = mdl.fit(gen_train,steps_per_epoch=steps_per_epoch,epochs=nb_epochs,callbacks=cbs, validation_data=(data_val.X,data_val.y),shuffle=True,initial_epoch=initial_epoch,use_multiprocessing=True,validation_freq=validation_freq)    # validation_data=(data_test.X,data_test.y)   validation_data=(data_val.X,data_val.y)   validation_batch_size=math.floor(n_val) # steps_per_epoch=steps_per_epoch

      
    rgb = mdl_history.history
    keys = list(rgb.keys())
    
    fname_history = 'history_'+mdl.name+'.h5'
    fname_history = os.path.join(path_model_base,fname_history)                            
    f = h5py.File(fname_history,'w')
    for i in range(len(rgb)):
        f.create_dataset(keys[i], data=rgb[keys[i]])
    f.close()
    
        
    return mdl_history


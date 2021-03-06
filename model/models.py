#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
""" 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Activation, Flatten, Reshape, MaxPool3D, MaxPool2D, Permute, BatchNormalization, GaussianNoise,DepthwiseConv2D, Dropout
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import math

 
# ----- HELPER FUNCS---- #
def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN_2D','PRFR_CNN2D','PR_CNN2D',
                 'BP_CNN2D_MULTIBP','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA','BP_CNN2D_MULTIBP_PRFRTRAINABLEGAMMA_RODS')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)

def get_model_memory_usage(batch_size, model):
    
    """ 
    Gets how much GPU memory will be required by the model.
    But doesn't work so good
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
    
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    
    return gbytes

def modelFileName(U=0,P=0,T=0,C1_n=0,C1_s=0,C1_3d=0,N_layers=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,BN=0,MP=0,LR=0,TR=0,with_TR=True):
    
    """
    Takes in data and model parameters, and parses them to 
    make the foldername where the model will be saved
    U : unit quality threshold
    P : Temporal dimension for the photoreceptor layer
    T : Temporal dimension for the first CNN
    C1, C2, C3 : CNN layers
    C_n : Number of channels in that CNN layer
    C_s : Convolution filter size (widht = height)
    C_3d : Size of the filter's third dimension in case CNNs are 3D Conv
    BN : BatchNormalization layer after each CNN (1=ON, 0=OFF)
    MP : MaxPool layer after first CNN (1=ON, 0=OFF)
    LR : Initial learning rate
    TR : Trial Number
    with_TR : Return filename with or without TR
    """
    
    def parse_param(key,val,fname):
        fname = fname+key+'-'+val+'_'    
        return fname

    fname = ''
    dict_params = {}
    
    U = '%0.2f'%U
    fname = parse_param('U',U,fname)    
    
    if P>0:
        P = '%03d' %P
        fname = parse_param('P',P,fname)    
    
    T = '%03d' %T
    fname = parse_param('T',T,fname)    
    
    if C1_3d>0:
        C1 = '%02d-%02d-%02d'%(C1_n,C1_s,C1_3d)
        dict_params['filt1_3rdDim'] = C1_3d
    else:
        C1 = '%02d-%02d'%(C1_n,C1_s)
        
    dict_params['chan1_n'] = C1_n
    dict_params['filt1_size'] = C1_s
    key = 'C1'
    fname = fname+key+'-'+eval(key)+'_'    
    
    if N_layers>0:
        N_layers = '%02d'%N_layers
        fname = parse_param('NL',N_layers,fname)    
        dict_params['N_layers'] = int(N_layers)

    if C2_n>0:
        if C2_3d>0:
            C2 = '%02d-%02d-%02d'%(C2_n,C2_s,C2_3d)
            dict_params['filt2_3rdDim'] = C2_3d
        else:
            C2 = '%02d-%02d'%(C2_n,C2_s)
        
        key = 'C2'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan2_n'] = C2_n
    dict_params['filt2_size'] = C2_s

        
    if C2_n>0 and C3_n>0:
        if C3_3d>0:
            C3 = '%02d-%02d-%02d'%(C3_n,C3_s,C3_3d)
            dict_params['filt3_3rdDim'] = C3_3d
        else:
            C3 = '%02d-%02d'%(C3_n,C3_s)
            
        key = 'C3'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan3_n'] = C3_n
    dict_params['filt3_size'] = C3_s
    
            
    BN = '%d'%BN
    fname = parse_param('BN',BN,fname)    
    dict_params['BatchNorm'] = int(BN)

    MP = '%d'%MP
    fname = parse_param('MP',MP,fname)    
    dict_params['MaxPool'] = int(MP)
    
    if LR>0:
        LR = '%0.4f'%LR
        fname = parse_param('LR',LR,fname)    
    
    if with_TR==True:
        TR = '%02d'%TR
        fname = parse_param('TR',TR,fname)    
        
    fname_model = fname[:-1]
    
    
    return fname_model,dict_params



def CNN2D(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])

    N_layers = 10
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    
    if chan1_n==1 and chan2_n<1:
        outputs = y
        
    else:

        if chan2_n>0:
            y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
            if BatchNorm is True:
                y = BatchNormalization(axis=-1)(y)
            y = Activation('relu')(y)
         
        if chan3_n>0:
            y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
            if BatchNorm is True:
                y = BatchNormalization(axis=-1)(y)
            y = Activation('relu')(y)
            
        # if chan3_n>0:
        #     for i in range(N_layers):
        #         y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        #         if BatchNorm is True:
        #             y = BatchNormalization(axis=-1)(y)
        #         y = Activation('relu')(y)
           
           
        y = Flatten()(y)
        if BatchNorm is True: 
            y = BatchNormalization()(y)
        y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
        outputs = Activation('softplus')(y)

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)

def CNN_DENSE(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    y = inputs
    
    # first layer  
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        rgb = y.shape
        y = Flatten()(y)
        y = BatchNormalization()(y)
        y = Reshape((rgb[1],rgb[2],rgb[3]))(y)
        
    y = Activation('relu')(y)
    
    if chan1_n==1 and chan2_n<1:
        outputs = y
        
    else:

        if N_layers>0:
            y = Flatten()(y)

            for i in range(N_layers):
                y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if BatchNorm is True:
                    y = BatchNormalization()(y)
                y = Activation('relu')(y)
                          
    # y = Flatten()(y)
    if BatchNorm is True:
        y = BatchNormalization()(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_DENSE'
    return Model(inputs, outputs, name=mdl_name)


# %% Adaptive-Conv
class Normalize_multichan(tf.keras.layers.Layer):
    """
    BatchNorm is where you calculate normalization factors for each dimension seperately based on
    the batch data
    LayerNorm is where you calculate the normalization factors based on channels and dimensions
    Normalize_multichan calculates normalization factors based on all dimensions for each channel seperately
    """
    
    def __init__(self,units=1):
        super(Normalize_multichan,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        inputs_perChan = tf.reshape(inputs,(-1,inputs.shape[-1]))
        value_min = tf.reduce_min(inputs_perChan,axis=0)
        value_max = tf.reduce_max(inputs_perChan,axis=0)
        
        # value_min = tf.expand_dims(value_min,axis=0)
        R_norm = (inputs - value_min[None,None,None,None,:])/(value_max[None,None,None,None,:]-value_min[None,None,None,None,:])
        R_norm_perChan = tf.reshape(R_norm,(-1,R_norm.shape[-1]))
        R_mean = tf.reduce_mean(R_norm_perChan,axis=0)       
        R_norm = R_norm - R_mean[None,None,None,None,:]
        return R_norm
    
def generate_simple_filter_multichan(tau,n,t):

    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    f = (t**n[:,None])*tf.math.exp(-t/tau[:,None]) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb[:,None])/tf.math.exp(tf.math.lgamma(n+1))[:,None] # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)
   
    return f

""" test filter 

    tau = tf.constant([[1]],dtype=tf.float32)
    n = tf.constant([[1]],dtype=tf.float32)
    t = tf.range(0,1000/timeBin,dtype='float32')
    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    tN = np.squeeze(t.eval(session=tf.compat.v1.Session()))
    a = t**n; aN = np.squeeze(a.eval(session=tf.compat.v1.Session()))
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb)/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    
    f = np.squeeze(f.eval(session=tf.compat.v1.Session()))
    plt.plot(f)


"""


def conv_oper_multichan(x,kernel_1D):
    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D,axis=0)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims,1])
    kernel_reshaped = tf.tile(kernel_1D,(tile_fac))
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,kernel_1D.shape[0],kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-2,0)
    pad_vec = [[0,0],[kernel_1D.shape[0]-1,0],[0,0],[0,0]]
    # pad_vec = [[0,0],[0,0],[0,0],[0,0]]
    conv_output = tf.nn.conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    # print(conv_output.shape)
    return conv_output

@tf.function
def slice_tensor(inp_tensor,shift_vals):
    # print(inp_tensor.shape)
    # print(shift_vals.shape)
    tens_reshape = tf.reshape(inp_tensor,[-1,inp_tensor.shape[1]*inp_tensor.shape[2]*inp_tensor.shape[3]*inp_tensor.shape[4]])
    shift_vals_new = ((inp_tensor.shape[1]-shift_vals[0,:])*shift_vals.shape[-1]) + tf.range(0,shift_vals.shape[-1])
    extracted_vals = tf.gather(tens_reshape,shift_vals_new,axis=1)
    extracted_vals_reshaped = tf.reshape(extracted_vals,(-1,1,inp_tensor.shape[2],inp_tensor.shape[3],inp_tensor.shape[4]))
    
    return extracted_vals_reshaped
    

    
    
# ADD CONSTRAINTS
class photoreceptor_DA_multichan_randinit(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_randinit,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
               
    def build(self,input_shape):    # random inits
    
        zeta_range = (0.00,0.01)
        zeta_init = tf.keras.initializers.RandomUniform(minval=zeta_range[0],maxval=zeta_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,zeta_range[0],zeta_range[1]))
        zeta_mulFac = tf.keras.initializers.Constant(1000.) 
        self.zeta_mulFac = self.add_weight(name='zeta_mulFac',initializer=zeta_mulFac,shape=[1,self.units],trainable=False)
        
        kappa_range = (0.00,0.01)
        kappa_init = tf.keras.initializers.RandomUniform(minval=kappa_range[0],maxval=kappa_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.kappa = self.add_weight(name='kappa',initializer=kappa_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,kappa_range[0],kappa_range[1]))
        kappa_mulFac = tf.keras.initializers.Constant(1000.) 
        self.kappa_mulFac = self.add_weight(name='kappa_mulFac',initializer=kappa_mulFac,shape=[1,self.units],trainable=False)
        
        alpha_range = (0.001,0.1)
        alpha_init = tf.keras.initializers.RandomUniform(minval=alpha_range[0],maxval=alpha_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(100.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.001,0.1)
        beta_init = tf.keras.initializers.RandomUniform(minval=beta_range[0],maxval=beta_range[1])  #tf.keras.initializers.Constant(0.02)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)

        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.RandomUniform(minval=gamma_range[0],maxval=gamma_range[1])  #tf.keras.initializers.Constant(0.075)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.001,0.02)
        tauY_init = tf.keras.initializers.RandomUniform(minval=tauY_range[0],maxval=tauY_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.1)
        nY_init = tf.keras.initializers.RandomUniform(minval=nY_range[0],maxval=nY_range[1]) #tf.keras.initializers.Constant(0.01)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


        tauZ_range = (0.3,1.)
        tauZ_init = tf.keras.initializers.RandomUniform(minval=tauZ_range[0],maxval=tauZ_range[1]) #tf.keras.initializers.Constant(0.5)# 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.01) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        tauC_range = (0.01,0.5)
        tauC_init = tf.keras.initializers.RandomUniform(minval=tauC_range[0],maxval=tauC_range[1])  #tf.keras.initializers.Constant(0.2)# 
        self.tauC = self.add_weight(name='tauC',initializer=tauC_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauC_range[0],tauC_range[1]))
        tauC_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauC_mulFac = tf.Variable(name='tauC_mulFac',initial_value=tauC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nC_range = (1e-5,0.5)
        nC_init = tf.keras.initializers.Constant(0.01) # tf.keras.initializers.RandomUniform(minval=nC_range[0],maxval=nC_range[1]) # 
        self.nC = self.add_weight(name='nC',initializer=nC_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nC_range[0],nC_range[1]))
        nC_mulFac = tf.keras.initializers.Constant(10.) 
        self.nC_mulFac = tf.Variable(name='nC_mulFac',initial_value=nC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha*self.alpha_mulFac
        beta = self.beta*self.beta_mulFac
        gamma =  self.gamma*self.gamma_mulFac
        zeta = self.zeta*self.zeta_mulFac
        kappa = self.kappa*self.kappa_mulFac
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        tau_c =  (self.tauC_mulFac*self.tauC) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        n_c =  (self.nC_mulFac*self.nC)
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kc = generate_simple_filter_multichan(tau_c,n_c,t)  
        Kz = generate_simple_filter_multichan(tau_z,n_z,t)  
        Kz = (gamma*Kc) + ((1-gamma) * Kz)
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
        
               
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        # print(z_tf_reshape.shape)
        
        y_shift = tf.math.argmax(Ky,axis=1);y_shift = tf.cast(y_shift,tf.int32)
        z_shift = tf.math.argmax(Kz,axis=1);z_shift = tf.cast(z_shift,tf.int32)
        
        y_tf_reshape = slice_tensor(y_tf_reshape,y_shift)
        z_tf_reshape = slice_tensor(z_tf_reshape,z_shift)
        # print(z_tf_reshape)
               
    
        # outputs = zeta[None,None,:,None,:] + (alpha[None,None,:,None,:]*y_tf_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        # outputs = zeta[None,None,0,None,:] + (alpha[None,None,0,None,:]*y_tf_reshape[:,:,0,:,:])/(1+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))
        outputs = (zeta[None,None,0,None,:] + (alpha[None,None,0,None,:]*y_tf_reshape[:,:,0,:,:]))/(kappa[None,None,0,None,:]+1e-6+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))
        # outputs = (alpha[None,None,0,None,:]*y_tf_reshape[:,:,0,:,:])/(kappa[None,None,0,None,:]+1e-6+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))
        # outputs = outputs[:,:,0,:,:]
        
        # print(outputs.shape)
        
        return outputs



def A_CNN_DENSE(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']
    
    if N_layers>2:
        N_half = int(chan2_n/2)
        rgb = (np.linspace(0,N_half,int(N_layers/2)+2)/1).astype('int32')
        rgb = rgb[1:]
        N_arr_dense = np.concatenate((rgb,np.flip(rgb[:-1])))
        N_arr_dense = N_arr_dense[:N_layers]



    
    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_randinit(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    # y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,0]       # only take the first time point
    
    # if BatchNorm==True:
    #     rgb = y.shape[1:]
    #     y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))


    if chan1_n==1 and chan2_n<1:
        y = Activation('softplus')(y)
        # y = Flatten()(y)
        # y = Activation('softplus')(y)
        # outputs = y
        
    else:
        y = Activation('relu')(y)
        
        if N_layers>0 and chan2_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                if N_layers>2:
                    # y = Dense(N_arr_dense[i], kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                else:
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if dropout_rate>0:
                    y = Dropout(dropout_rate)(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
    # Dense layer
    y = Flatten()(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN_DENSE'
    return Model(inputs, outputs, name=mdl_name)


def ACNN_NORM_FLATCNN(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']


    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_randinit(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    # y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,0]       # only take the first time point
    
    
    y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    

    if chan1_n==1 and chan2_n<1:
        pass   
     
    else:
        # y = Activation('relu')(y)
        
        if N_layers>0 and chan2_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                if N_layers>2:
                    # y = Dense(N_arr_dense[i], kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                else:
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if dropout_rate>0:
                    y = Dropout(dropout_rate)(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
    # Dense layer
    y = Flatten()(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'ACNN_NORM_FLATCNN'
    return Model(inputs, outputs, name=mdl_name)

# %% A_CNN as NORM LAYER
class photoreceptor_DA_multichan_norm(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_norm,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
               
    def build(self,input_shape):    # random inits
    
        zeta_range = (0.00,0.1)
        zeta_init = tf.keras.initializers.RandomUniform(minval=zeta_range[0],maxval=zeta_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,zeta_range[0],zeta_range[1]))
        zeta_mulFac = tf.keras.initializers.Constant(100.) 
        self.zeta_mulFac = self.add_weight(name='zeta_mulFac',initializer=zeta_mulFac,shape=[1,self.units],trainable=False)

        alpha_range = (0.001,0.1)
        alpha_init = tf.keras.initializers.RandomUniform(minval=alpha_range[0],maxval=alpha_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(100.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.001,0.1)
        beta_init = tf.keras.initializers.RandomUniform(minval=beta_range[0],maxval=beta_range[1])  #tf.keras.initializers.Constant(0.02)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)

        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.RandomUniform(minval=gamma_range[0],maxval=gamma_range[1])  #tf.keras.initializers.Constant(0.075)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.001,0.02)
        tauY_init = tf.keras.initializers.RandomUniform(minval=tauY_range[0],maxval=tauY_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.1)
        nY_init = tf.keras.initializers.RandomUniform(minval=nY_range[0],maxval=nY_range[1]) #tf.keras.initializers.Constant(0.01)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


        tauZ_range = (0.3,1.)
        tauZ_init = tf.keras.initializers.RandomUniform(minval=tauZ_range[0],maxval=tauZ_range[1]) #tf.keras.initializers.Constant(0.5)# 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.01) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        tauC_range = (0.01,0.5)
        tauC_init = tf.keras.initializers.RandomUniform(minval=tauC_range[0],maxval=tauC_range[1])  #tf.keras.initializers.Constant(0.2)# 
        self.tauC = self.add_weight(name='tauC',initializer=tauC_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauC_range[0],tauC_range[1]))
        tauC_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauC_mulFac = tf.Variable(name='tauC_mulFac',initial_value=tauC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nC_range = (1e-5,0.5)
        nC_init = tf.keras.initializers.Constant(0.01) # tf.keras.initializers.RandomUniform(minval=nC_range[0],maxval=nC_range[1]) # 
        self.nC = self.add_weight(name='nC',initializer=nC_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nC_range[0],nC_range[1]))
        nC_mulFac = tf.keras.initializers.Constant(10.) 
        self.nC_mulFac = tf.Variable(name='nC_mulFac',initial_value=nC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha*self.alpha_mulFac
        beta = self.beta*self.beta_mulFac
        gamma =  self.gamma*self.gamma_mulFac
        zeta = self.zeta*self.zeta_mulFac
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        tau_c =  (self.tauC_mulFac*self.tauC) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        n_c =  (self.nC_mulFac*self.nC)
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kc = generate_simple_filter_multichan(tau_c,n_c,t)  
        Kz = generate_simple_filter_multichan(tau_z,n_z,t)  
        Kz = (gamma*Kc) + ((1-gamma) * Kz)
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
        
               
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],tau_y.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        # print(z_tf_reshape.shape)
        
        y_shift = tf.math.argmax(Ky,axis=1);y_shift = tf.cast(y_shift,tf.int32)
        z_shift = tf.math.argmax(Kz,axis=1);z_shift = tf.cast(z_shift,tf.int32)
        
        y_tf_reshape = slice_tensor(y_tf_reshape,y_shift)
        z_tf_reshape = slice_tensor(z_tf_reshape,z_shift)
        # print(z_tf_reshape)
        
        # inputs_reshape = inputs[:,-2:-1,:,:]
        
        z_tf_new = z_tf_reshape[:,:,0,:,:]
    
        # outputs = zeta[None,None,:,None,:] + (alpha[None,None,:,None,:]*inputs[:,:,:,None,None])/(1+(beta[None,None,0,None,:]*z_tf_new[:,None,:,:,:]))
        outputs = (alpha[None,None,:,None,:]*inputs[:,:,:,None,None])/(1+(beta[None,None,0,None,:]*z_tf_new[:,None,:,:,:]))
        # outputs = outputs[:,:,0,:,:]
        
        # print(outputs.shape)
        
        return outputs



def A_CNN_NORM(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']
    
    if N_layers>2:
        N_half = int(chan2_n/2)
        rgb = (np.linspace(0,N_half,int(N_layers/2)+2)/1).astype('int32')
        rgb = rgb[1:]
        N_arr_dense = np.concatenate((rgb,np.flip(rgb[:-1])))
        N_arr_dense = N_arr_dense[:N_layers]



    chan_bp = 1
    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_norm(units=chan_bp,kernel_regularizer=l2(1e-4))(y)
    y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan_bp))(y)
    # y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan_bp))(y)
    # y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,10:,:,:,0]       # remove first few points for any boundary effects
    
    # if BatchNorm==True:
    #     rgb = y.shape[1:]
    #     y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
    
    
    # CNN layer  
    filt1_size = y.shape[2]
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)


    if chan1_n==1 and chan2_n<1:
        outputs = y
    else:
        y = Activation('relu')(y)
        
        if N_layers>0 and chan2_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                if N_layers>2:
                    # y = Dense(N_arr_dense[i], kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                else:
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if dropout_rate>0:
                    y = Dropout(dropout_rate)(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
        # Dense layer
        y = Flatten()(y)
    
        if BatchNorm is True: 
            y = BatchNormalization(axis=-1)(y)
        y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
        outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN_NORM'
    return Model(inputs, outputs, name=mdl_name)


# %% A_CNN followed by 3D CNNs
class photoreceptor_DA_multichan_comb(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_comb,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
               
    def build(self,input_shape):    # random inits
    
        zeta_range = (0.00,0.1)
        zeta_init = tf.keras.initializers.RandomUniform(minval=zeta_range[0],maxval=zeta_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,zeta_range[0],zeta_range[1]))
        zeta_mulFac = tf.keras.initializers.Constant(100.) 
        self.zeta_mulFac = self.add_weight(name='zeta_mulFac',initializer=zeta_mulFac,shape=[1,self.units],trainable=False)

        alpha_range = (0.001,0.1)
        alpha_init = tf.keras.initializers.RandomUniform(minval=alpha_range[0],maxval=alpha_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(100.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.001,0.1)
        beta_init = tf.keras.initializers.RandomUniform(minval=beta_range[0],maxval=beta_range[1])  #tf.keras.initializers.Constant(0.02)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)

        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.RandomUniform(minval=gamma_range[0],maxval=gamma_range[1])  #tf.keras.initializers.Constant(0.075)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.001,0.02)
        tauY_init = tf.keras.initializers.RandomUniform(minval=tauY_range[0],maxval=tauY_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.1)
        nY_init = tf.keras.initializers.RandomUniform(minval=nY_range[0],maxval=nY_range[1]) #tf.keras.initializers.Constant(0.01)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


        tauZ_range = (0.3,1.)
        tauZ_init = tf.keras.initializers.RandomUniform(minval=tauZ_range[0],maxval=tauZ_range[1]) #tf.keras.initializers.Constant(0.5)# 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.01) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        tauC_range = (0.01,0.5)
        tauC_init = tf.keras.initializers.RandomUniform(minval=tauC_range[0],maxval=tauC_range[1])  #tf.keras.initializers.Constant(0.2)# 
        self.tauC = self.add_weight(name='tauC',initializer=tauC_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauC_range[0],tauC_range[1]))
        tauC_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauC_mulFac = tf.Variable(name='tauC_mulFac',initial_value=tauC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nC_range = (1e-5,0.5)
        nC_init = tf.keras.initializers.Constant(0.01) # tf.keras.initializers.RandomUniform(minval=nC_range[0],maxval=nC_range[1]) # 
        self.nC = self.add_weight(name='nC',initializer=nC_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nC_range[0],nC_range[1]))
        nC_mulFac = tf.keras.initializers.Constant(10.) 
        self.nC_mulFac = tf.Variable(name='nC_mulFac',initial_value=nC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs_call):
       
        timeBin = 1
        
        alpha =  self.alpha*self.alpha_mulFac
        beta = self.beta*self.beta_mulFac
        gamma =  self.gamma*self.gamma_mulFac
        zeta = self.zeta*self.zeta_mulFac
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        tau_c =  (self.tauC_mulFac*self.tauC) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        n_c =  (self.nC_mulFac*self.nC)
        
        t = tf.range(0,inputs_call.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kc = generate_simple_filter_multichan(tau_c,n_c,t)  
        Kz = generate_simple_filter_multichan(tau_z,n_z,t)  
        Kz = (gamma*Kc) + ((1-gamma) * Kz)
        
        y_tf = conv_oper_multichan(inputs_call,Ky)
        z_tf = conv_oper_multichan(inputs_call,Kz)
        
               
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs_call.shape[-1],tau_y.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs_call.shape[-1],tau_z.shape[-1]))
        # print(z_tf_reshape.shape)
        
        # y_shift = tf.math.argmax(Ky,axis=1);y_shift = tf.cast(y_shift,tf.int32)
        # z_shift = tf.math.argmax(Kz,axis=1);z_shift = tf.cast(z_shift,tf.int32)
        
        # y_tf_reshape = slice_tensor(y_tf_reshape,y_shift)
        # z_tf_reshape = slice_tensor(z_tf_reshape,z_shift)
        # print(z_tf_reshape)
        
        # inputs_reshape = inputs[:,-2:-1,:,:]
        
        z_tf_new = z_tf_reshape[:,:,0,:,:]
    
        # outputs = zeta[None,None,:,None,:] + (alpha[None,None,:,None,:]*y_tf_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        # outputs = outputs[:,:,0,:,:]
        # outputs = zeta[None,None,:,None,:] + (alpha[None,None,:,None,:]*inputs[:,:,:,None,None])/(1+(beta[None,None,0,None,:]*z_tf_new[:,None,:,:,:]))
        outputs = zeta[None,None,0,None,:] + (alpha[None,None,0,None,:]*inputs_call[:,:,:,None])/(1+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))

        # print(outputs.shape)
        
        return outputs



def ACNN_NORM_3DCNN(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']

    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_comb(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = y[:,10:,:,:,:]       # remove first few points for any boundary effects
    
    
    y = Conv3D(chan2_n, (y.shape[1],filt2_size,filt2_size), data_format="channels_last", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)


    if chan1_n==1 and chan2_n<1:
        
        pass
        
    else:
        y = Activation('relu')(y)
        
        if N_layers>0 and chan3_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                y = Dense(chan3_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if dropout_rate>0:
                    y = Dropout(dropout_rate)(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
        # Dense layer
        y = Flatten()(y)
    
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'ACNN_NORM_3DCNN'
    return Model(inputs, outputs, name=mdl_name)


def ACNN_NORM_2DCNN(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']

    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_comb(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = y[:,10:,:,:,:]       # remove first few points for any boundary effects
    
    
    # CNN layer  

    y = Reshape((y.shape[1],y.shape[2]*y.shape[3],y.shape[4]))(y)
    y = Conv2D(1, [1,1], data_format="channels_last", kernel_regularizer=l2(1e-3))(y)
    
    y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)


    if chan1_n==1 and chan2_n<1:
        
        pass
        
    else:
        y = Activation('relu')(y)
        
        if N_layers>0 and chan3_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                y = Dense(chan3_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if dropout_rate>0:
                    y = Dropout(dropout_rate)(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
    # Dense layer
    y = Flatten()(y)
    
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'ACNN_NORM_2DCNN'
    return Model(inputs, outputs, name=mdl_name)


def ACNN_CNNS(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers']
    dropout_rate = kwargs['dropout']

    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    y = photoreceptor_DA_multichan_comb(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    y = Reshape((inputs.shape[-3],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = y[:,10:,:,:,:]       # remove first few points for any boundary effects
    
    
    # CNN layer  

    y = Reshape((y.shape[1],y.shape[2]*y.shape[3],y.shape[4]))(y)
    y = Conv2D(1, [1,1], data_format="channels_last", kernel_regularizer=l2(1e-3))(y)
    
    y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)


    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
            
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)

        
    # Dense layer
    y = Flatten()(y)
    
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'ACNN_CNNS'
    return Model(inputs, outputs, name=mdl_name)

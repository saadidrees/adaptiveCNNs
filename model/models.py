#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
"""
 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Activation, Flatten, Reshape, MaxPool3D, MaxPool2D, Permute, BatchNormalization, GaussianNoise,DepthwiseConv2D
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

def modelFileName(U=0,P=0,T=0,C1_n=0,C1_s=0,C1_3d=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,BN=0,MP=0,LR=0,TR=0,with_TR=True):
    
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

# %% Standard models
def CNN2D_DW(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    filt_temporal_width=kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])

    
    sigma = 0.1
    

    # first layer  
    if BatchNorm is True:
        y = inputs
        y = DepthwiseConv2D(depth_multiplier=chan1_n, kernel_size=filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        y = BatchNormalization(axis=-1)(y)
    else:
        y = DepthwiseConv2D(depth_multiplier=chan1_n, kernel_size=filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]

    # y = Activation('relu')(y) #(GaussianNoise(sigma)(y))
    # outputs = Activation('softplus')(y)
    y = y[:,-1,:,:]
    y = Flatten()(y)
    outputs = Activation('softplus')(y)
    # outputs = y


    # second layer
    if chan2_n>0:
        y = DepthwiseConv2D(depth_multiplier=chan2_n, kernel_size=filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        # y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  

        if BatchNorm is True: 
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        y = Activation('relu')(y) #(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)       

        if BatchNorm is True: 
            
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(y) #(GaussianNoise(sigma)(y))
        
        
    # y = Flatten()(y)
    # if BatchNorm is True: 
    #     y = BatchNormalization(axis=-1)(y)
    # y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    # outputs = Activation('softplus')(y)

    mdl_name = 'CNN2D_DW'
    return Model(inputs, outputs, name=mdl_name)


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
            
        if chan3_n>0:
            for i in range(N_layers):
                y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
                if BatchNorm is True:
                    y = BatchNormalization(axis=-1)(y)
                y = Activation('relu')(y)
           
           
        y = Flatten()(y)
        if BatchNorm is True: 
            y = BatchNormalization()(y)
        y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
        outputs = Activation('softplus')(y)

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)

def CNN2D_DENSE(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
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

    # first layer  
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    if BatchNorm is True:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    
    if chan1_n==1 and chan2_n<1:
        outputs = y
        
    else:

        if N_layers>0:
            for i in range(N_layers):
                y = Flatten()(y)
                if BatchNorm is True: 
                    y = BatchNormalization()(y)
                y = Dense(chan2_n, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
                y = Activation('relu')(y)
                          
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization()(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN2D_DENSE'
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
    # pad_vec = [[0,0],[kernel_1D.shape[0]-1,0],[0,0],[0,0]]
    pad_vec = [[0,0],[0,0],[0,0],[0,0]]
    # conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    conv_output = tf.nn.conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    # print(conv_output.shape)
    return conv_output

class photoreceptor_DA_multichan_randinit(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_randinit,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
               
    def build(self,input_shape):    # random inits
               
        alpha_init = tf.keras.initializers.RandomUniform(1.) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        beta_init = tf.keras.initializers.RandomUniform(1.) 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        gamma_init = tf.keras.initializers.RandomUniform(1.)
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        zeta_init = tf.keras.initializers.RandomUniform(1.)
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        tauY_init = tf.keras.initializers.RandomUniform(1.) # 0.5
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        tauY_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)

        tauZ_init = tf.keras.initializers.RandomUniform(1.) 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        tauZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        nY_init = tf.keras.initializers.RandomUniform(1.) 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        nZ_init = tf.keras.initializers.RandomUniform(1.) 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha
        beta = self.beta
        gamma =  self.gamma
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        zeta = self.zeta
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter_multichan(tau_z,n_z,t))
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
                
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
    
        outputs = (alpha[None,None,:,None,:]*y_tf_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        # outputs = (alpha[None,None,:,None,:]*inputs_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        
        outputs = outputs[:,:,0,:,:]
        
        return outputs



class photoreceptor_DA_multichan(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
    def build(self,input_shape):
               
        alpha_init = tf.keras.initializers.Constant(1.) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        beta_init = tf.keras.initializers.Constant(0.) 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        gamma_init = tf.keras.initializers.Constant(0.5)
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        tauY_init = tf.keras.initializers.Constant(0.1) # 0.5
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        tauY_mulFac = tf.keras.initializers.Constant(10.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)

        tauZ_init = tf.keras.initializers.Constant(0.5) 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        tauZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        nY_init = tf.keras.initializers.Constant(1.) 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        nY_mulFac = tf.keras.initializers.Constant(1.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        nZ_init = tf.keras.initializers.Constant(1.) 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer)
        
        nZ_mulFac = tf.keras.initializers.Constant(1.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha
        beta = self.beta
        gamma =  self.gamma
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter_multichan(tau_z,n_z,t))
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
                
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],alpha.shape[-1]))
    
        outputs = (alpha[None,None,:,None,:]*y_tf_reshape)/(1+(beta[None,None,:,None,:]*z_tf_reshape))
        
        outputs = outputs[:,:,0,:,:]
        
        return outputs



def A_CNN(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_DA_multichan_randint(units=chan1_n,kernel_regularizer=l2(1e-6))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,-1]       # only take the last time point
    
    if BatchNorm==True:
        y = BatchNormalization()(y)
    
    if chan1_n==1 and chan2_n<1:
        y = Flatten()(y)
        outputs = Activation('softplus')(y)
    else:
        y = Activation('relu')(y)

        # CNN - first layer
        if chan2_n>0:
            y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-4))(y)
            if BatchNorm is True:
                y = BatchNormalization()(y)
            y = Activation('relu')(y)

        if chan3_n>0:
            y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-4))(y)
            if BatchNorm is True:
                y = BatchNormalization()(y)
            y = Activation('relu')(y)

        
        # Dense layer
        y = Flatten()(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=-1)(y)
        y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
        outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN'
    return Model(inputs, outputs, name=mdl_name)

def A_CNN_DENSE(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    N_layers = kwargs['N_layers'];
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_DA_multichan_randinit(units=chan1_n,kernel_regularizer=l2(1e-5))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,-1]       # only take the last time point
    
    if BatchNorm==True:
        y = BatchNormalization()(y)
    
    if chan1_n==1 and chan2_n<1:
        # y = Activation('softplus')(y)
        # outputs = y
        y = Activation('relu')(y)
        
    else:
        y = Activation('relu')(y)

        if N_layers>0 and chan2_n>0:
            for i in range(N_layers):
                y = Flatten()(y)
                if BatchNorm is True: 
                    y = BatchNormalization(axis=-1)(y)
                y = Dense(chan2_n, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                y = Activation('softplus')(y)

        
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN_DENSE'
    return Model(inputs, outputs, name=mdl_name)



def A_CNN_MULTILAYER(inputs,n_out,**kwargs): # BP --> 3D CNN --> 2D CNN
    
    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']; filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']; filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_DA_multichan(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    # y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    # y = y[:,inputs.shape[1]-filt_temporal_width:,:,:,:]
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,-1]       # only take the last time point

    
    if BatchNorm==True:
        y = BatchNormalization()(y)
    
    if chan1_n==1 and chan2_n<1:
        # y = Flatten()(y)
        # outputs = Activation('softplus')(y)
        y = Activation('relu')(y)
    else:
        y = Activation('relu')(y)


        # CNN - first layer
        if chan2_n>0:
            rgb = y
            y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]))(y)
            y = photoreceptor_DA_multichan_randint(units=chan2_n,kernel_regularizer=l2(1e-4))(y)
            y = Reshape((1,rgb.shape[-2],rgb.shape[-1],chan2_n))(y)
            y = Permute((4,2,3,1))(y)   # Channels first
            y = y[:,:,:,:,-1]       # only take the last time point
            if BatchNorm is True:
                y = BatchNormalization()(y)
            y = Activation('relu')(y)
        
        if chan3_n>0:
            y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-4))(y)
            if BatchNorm is True:
                y = BatchNormalization()(y)
            y = Activation('relu')(y)

        
        # Dense layer
        y = Flatten()(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=-1)(y)
        y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
        outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN'
    return Model(inputs, outputs, name=mdl_name)



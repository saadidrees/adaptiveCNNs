#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:40:36 2022

@author: saad
""" 
import numpy as np
import re
import os

def get_weightsDict(mdl):
    names = [weight.name for layer in mdl.layers for weight in layer.weights]
    weights = mdl.get_weights()
    weights_dict = {}
    for i in range(len(names)):
        weight_name = names[i][:-2]        
        weights_dict[weight_name] = np.atleast_1d(np.squeeze(weights[i]))
        
    return weights_dict

def get_weightsOfLayer(weights_dict,layer_name):
    weights_keys = list(weights_dict.keys())
    rgb = re.compile(layer_name+'/')
    layer_weight_names = list(filter(rgb.match, weights_keys))
    weights_layer = {}
    for l_name in layer_weight_names:
        param_name_full = os.path.basename(l_name)
        rgb = re.findall(r'[^0-9]',param_name_full)
        rgb = ''.join(rgb)
        if rgb[-1] == '_':
            rgb = rgb[:-1]
        
        param_name = rgb
        weights_layer[param_name] = weights_dict[l_name]
    
    return weights_layer

def get_gradsDict(mdl,grads):
    variables_names = [v.name for v in mdl.trainable_variables]
    grads_dict = {}
    grads_dict_nans = {}
    for i in range(len(variables_names)):
        var_name = variables_names[i][:-2]        
        grads_dict[var_name] = np.atleast_1d(np.squeeze(grads[i]))
        grads_dict_nans[var_name] = np.sum(np.isnan(grads_dict[var_name]))
        
    return grads_dict,grads_dict_nans

def get_gradientsOfVar(grads_dict,layer_name_select):
    grad_keys = list(grads_dict.keys())
    rgb = re.compile(layer_name_select+'/')
    layer_weight_names = list(filter(rgb.match, grad_keys))
    grads_layer = {}
    for l_name in layer_weight_names:
        param_name_full = os.path.basename(l_name)
        rgb = re.findall(r'[^0-9]',param_name_full)
        rgb = ''.join(rgb)
        if rgb[-1] == '_':
            rgb = rgb[:-1]
        
        param_name = rgb
        grads_layer[param_name] = grads_dict[l_name]
    
    return grads_layer

        
def get_layerNames(mdl):
    layer_names = [layer.name for layer in mdl.layers]
    return layer_names


def updateWeights_allLayers(weights_dict,grads_dict,lr):
    weights_dict_updated = {}
    weights_dict_nans = {}
    for v_name in grads_dict.keys():
        updatedWeights = weights_dict[v_name] - (lr*grads_dict[v_name])
        weights_dict_updated[v_name] = updatedWeights
        weights_dict_nans[v_name] = np.sum(np.isnan(updatedWeights))
        
    return weights_dict_updated,weights_dict_nans
    
    
    
    
    
    
    
    
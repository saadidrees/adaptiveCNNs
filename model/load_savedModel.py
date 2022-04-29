#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:09:17 2021

@author: saad
"""
from model import activations, metrics
from tensorflow.keras.models import load_model

def load(fname_model):
    """Reload a keras model"""
    objects = {k: activations.__dict__[k] for k in activations.__all__}
    objects.update({k: metrics.__dict__[k] for k in metrics.__all__})
    return load_model(fname_model, custom_objects=objects)

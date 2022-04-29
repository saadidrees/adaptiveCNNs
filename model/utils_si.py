#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:34:40 2021

@author: saad
"""

import numpy as np
import math

def h5_tostring(arr):
    new_arr = np.empty(arr.shape[0],dtype='object')
    for i in range(arr.shape[0]):
        rgb = arr[i].tostring().decode('ascii')
        rgb = rgb.replace("\x00","")
        new_arr[i] = rgb

    return (new_arr)
    
def splitall(path):
    import os, sys

    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

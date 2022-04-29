#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:03:37 2022

@author: saad
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import lfilter
from scipy.signal import deconvolve
from scipy import integrate


def generate_simple_filter(tau,n,t):
    f = (t**n)*np.exp(-t/tau) # functional form in paper
    f = (f/tau**(n+1))/np.exp(math.lgamma(n+1)) # normalize appropriately

    return f

def DA_model(stim,params):
       
    def dxdt(t,x,params_ode):
        b = params_ode['beta'] 
        a = params_ode['alpha'] 
        tau_r = params_ode['tau_r'] 
        y = params_ode['y'] 
        z = params_ode['z'] 
        
        zt = np.interp(t,np.arange(0,z.shape[0]),z)
        yt = np.interp(t,np.arange(0,y.shape[0]),y)
        
        dx = (1/tau_r) * ((a*yt) - ((1+(b*zt))*x))
        
        return dx

    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    tau_y = params['tau_y']
    n_y = params['n_y']   
    tau_z = params['tau_z']
    n_z = params['n_z']
    tau_r = params['tau_r']
    
    
    t = np.arange(0,3000/timeBin)

    Ky = generate_simple_filter(tau_y,n_y,t)
    Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))

    # y = lfilter(Ky,1,stim)
    # z = lfilter(Kz,1,stim)

    y = np.convolve(stim,Ky)
    z = np.convolve(stim,Kz)

    if tau_r > 0:
        params_ode = {}
        params_ode['alpha'] = alpha
        params_ode['beta'] = beta
        params_ode['y'] = y
        params_ode['z'] = z
        params_ode['tau_r'] = tau_r
        
        T0 = np.array([1,z.shape[0]])
        X0 = 0
        
        ode15s = integrate.ode(dxdt).set_integrator('vode', method='bdf', order=5, max_step=25)
        ode15s.set_initial_value(X0).set_f_params(params_ode)
        dt = 10
        R = np.atleast_1d(0)
        T = np.atleast_1d(0)
        while ode15s.successful() and ode15s.t < T0[-1]:
            ode15s.integrate(ode15s.t+dt)
            R = np.append(R,ode15s.y)
            T = np.append(T,ode15s.t)        
        R = R[1:]
        T = T[1:]
        
        if dt>1:
            R = np.interp(np.arange(0,z.shape[0]),T,R)

    else:   
        R = alpha*y/(1+(beta*z))
        
    return Ky,Kz,y,z,R

def DA_model_iter(stim,params):
    
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    tau_y = params['tau_y']
    n_y = params['n_y']   
    tau_z = params['tau_z']
    n_z = params['n_z']
    tau_r = params['tau_r']
    
        
    t = np.ceil(np.arange(0,3000/timeBin))

    Ky = generate_simple_filter(tau_y,n_y,t)
    Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))

    y = np.convolve(stim,Ky)
    z = np.convolve(stim,Kz)
   
    
    if tau_r > 0:
        
        zt = np.zeros((z.shape[0]))
        yt = np.zeros((z.shape[0]))
        R = np.zeros((z.shape[0]))
        
        zt[0] = z[0]
        yt[0] = y[0]
        R[0] = alpha*yt[0]/(1+(beta*zt[0]))
        
        for pnt in range(1,z.shape[0]):
            zt[pnt] = z[pnt] #np.interp(pnt,np.arange(0,z.shape[0]),z)
            yt[pnt] = y[pnt] #np.interp(pnt,np.arange(0,y.shape[0]),y)
            
            dx = (1/tau_r) * ((alpha*yt[pnt]) - ((1+(beta*zt[pnt]))*R[pnt-1]))
            R[pnt] = R[pnt-1] + dx
        
    else:   
        R = alpha*y/(1+(beta*z))
        
        
    return Ky,Kz,y,z,R


timeBin = 10
stim = np.random.rand(100)
stim = np.repeat(stim,timeBin)

# %%
tau_y = 5
n_y = 5
tau_z = 100
n_z = 5
tau_r = 10

alpha = 1
beta = 1
gamma = 0.5

params = dict(tau_y=tau_y,n_y=n_y,tau_z=tau_z,n_z=n_z,alpha=alpha,beta=beta,gamma=gamma,tau_r=tau_r)

# t = np.arange(0,3000/timeBin)

# Ky = generate_simple_filter(tau_y,n_y,t)   
# Kz = (gamma*Ky) + ((1-gamma) * generate_simple_filter(tau_z,n_z,t))
# filt_len = Ky.shape[0]+1
   
# y = np.convolve(stim,Ky)
# z = np.convolve(stim,Kz)
# outputs_clark = (alpha*y)/(1+(beta*z))

Ky,Kz,y,z,outputs_clark = DA_model(stim,params)

plt.plot(Ky)
plt.plot(Kz)
plt.show()

plt.plot(y[:-filt_len])#/y.max())
plt.plot(z[:-filt_len])#/z.max())
plt.plot(outputs_clark[:-filt_len])#/outputs_clark.max())
plt.show()


# %%


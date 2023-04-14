#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:03:28 2022

@author: do0236li
"""
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### simulation x with given initial condition and \bar{sigma} #####
def simulate_x_with_sigma(model_fxs, x_init, s_init, dt_x, idx_init, s1, length):
    x_series = [x_init.flatten()]
    s_series = [s_init.flatten()]
    t_series = [np.array([0])]
    fx_series = []

    for i in range(idx_init, idx_init+length):
        dt = dt_x[i]
        txs = torch.tensor([np.r_[np.array([1]),x_series[-1],s_series[-1]]], dtype=torch.float32).to(device)
        f_numerical = model_fxs(txs).cpu().detach().numpy()
        
        fx_numerical = f_numerical[:,:-1]
        xx_ = x_series[-1] + fx_numerical*dt ##Euler method
        x_series.append(xx_.flatten())
        
        ss_ = s1[i]   ###using real ss_
        s_series.append(ss_)
        
        t_series.append(t_series[-1]+dt)
        fx_series.append(fx_numerical.flatten())
        
    x_series = np.array(x_series)[1:]
    s_series = np.array(s_series)[1:]
    t_series = np.array(t_series)[1:]
    fx_series = np.array(fx_series)
    
    return x_series, s_series, t_series, fx_series


### simulate both x and sigma with Gaussian noise
def simulate_xs_with_GN(model_f, x_init, s_init, dt, length, cond_std, cond_mean=None, x_inter=None, s_inter=None, stop_threshold=-20):
    
    x_series, s_series, t_series = x_init, s_init, [np.array([0])]
    for i in range(length-1):
        txs = torch.tensor([np.r_[np.array([1]),x_series[-1],s_series[-1]]], dtype=torch.float32).to(device)
        f_numerical = model_f(txs).cpu().detach().numpy()
        
        fx_numerical = f_numerical[:,:-1]
        xx_ = x_series[-1] + fx_numerical*dt ##Euler method
        x_series.append(xx_.flatten())
        
        try: ##if cond_std is not scalar
            bins_x, bins_s = cond_std.shape
            ### conditional std ###
            x_idx = (x_series[-1][0]>x_inter).sum()-1
            s_idx = (s_series[-1][0]>s_inter).sum()-1
            x_idx = np.clip(x_idx, 0, bins_x-1)
            s_idx = np.clip(s_idx, 0, bins_s-1)

            std = cond_std[x_idx, s_idx]
            mean = cond_mean[x_idx, s_idx]

        except: ##if cond_std is scalar
            std = cond_std
            mean = 0
            
        fs_numerical = f_numerical[:,[-1]] 
        ss_ = s_series[-1] + fs_numerical*dt + (np.random.randn()*std+mean)*dt**.5 ##Euler method, consider noise is Brownian motion
        s_series.append(ss_.flatten())
        
        t_series.append(t_series[-1]+dt)
        
        if xx_.flatten()[0]<stop_threshold:
            break
        
    x_series, s_series, t_series = np.array(x_series), np.array(s_series), np.array(t_series)
    return x_series, s_series, t_series


### simulate both x and sigma with Gaussian noise, std and mean by NN
def simulate_xs_with_GNN(model_f, model_std_mean, x_init, s_init, dt, length, stop_threshold=-20):
    
    x_series, s_series, t_series = x_init, s_init, [np.array([0])]
    for i in range(length-1):
        txs = torch.tensor(np.array([np.r_[np.array([1]),x_series[-1],s_series[-1]]]), dtype=torch.float32).to(device)
        f_numerical = model_f(txs).cpu().detach().numpy()
        
        fx_numerical = f_numerical[:,:-1]
        xx_ = x_series[-1] + fx_numerical*dt ##Euler method
        x_series.append(xx_.flatten())
        
        std_mean = model_std_mean(txs).cpu().detach().numpy()
        std, mean = std_mean[:,0], std_mean[:,1]
        
        fs_numerical = f_numerical[:,[-1]] 
        ss_ = s_series[-1] + fs_numerical*dt + (np.random.randn()*std+mean)*dt**.5 ##Euler method, consider noise is Brownian motion
        s_series.append(ss_.flatten())
        
        t_series.append(t_series[-1]+dt)
        
        if xx_.flatten()[0]<stop_threshold: ##for saddle data
            break
    x_series, s_series, t_series = np.array(x_series), np.array(s_series), np.array(t_series)
    return x_series, s_series, t_series


### simulate both x and sigma with approximated empirical noise(model_eps)
def simulate_xs_with_EN(model_f, model_eps, x_init, s_init, dt=0.01, length=10000, stop_threshold=-20, lamb=.5):
    x_series, s_series, t_series = x_init, s_init, [np.array([0])]
     # dt is larger than dt in given data
    for i in range(length-1):
        txs = torch.tensor([np.r_[np.array([1]),x_series[-1],s_series[-1]]], dtype=torch.float32).to(device)
        f_numerical = model_f(txs).cpu().detach().numpy()
        
        fx_numerical = f_numerical[:,:-1]
        xx_ = x_series[-1] + fx_numerical*dt ##Euler method
        x_series.append(xx_.flatten())
    
        obs = torch.tensor([np.r_[x_series[-1],s_series[-1], np.random.rand()]], dtype=torch.float32).to(device)
        eps = model_eps(obs).cpu().detach().numpy().item()
        
        fs_numerical = f_numerical[:,[-1]]
        # fs_numerical = fs_np(1,xx_,ss_)
        ss_ = s_series[-1] + fs_numerical*dt + eps*dt**lamb ##Euler method
        s_series.append(ss_.flatten())
        
        t_series.append(t_series[-1]+dt)
            
        if xx_.flatten()[0]<stop_threshold: ##for saddle data
            break
        
    x_series = np.array(x_series)
    s_series = np.array(s_series)
    t_series = np.array(t_series)
    return x_series, s_series, t_series
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:44:14 2022

@author: dliu
"""

import sys 
sys.path.append("..") 
import torch
import time
import numpy as np
from utils import get_dataset, get_inter_grid, get_cond_noise
from misc import train_model, parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10, lr=5e-4)

if __name__=="__main__":
    
    data_train_pathes = params.data_pathes[:params.n_trainset]
    ### load data ###
    x_lower_bound = params.x_lower_bound
    kx = params.kx
    ks = params.ks
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)

    

    #####################
    ### train model N ###
    #####################
    from model import model_f_hopf
    model_f = model_f_hopf().to(device)
    print(f'model N has {count_parameters(model_f)} parameters')

    inputs = np.c_[np.ones_like(s1), x1, s1]
    target = np.c_[diff_x,diff_s]
    t0 = time.time()
    model_f = train_model(model_f, inputs, target, niter=20000, lr=params.lr)
    print(f'time for training model f: {(time.time()-t0)/60}')

    ### save model
    torch.save(model_f.state_dict(), params.model_f_path) 
    
    
    
    #####################
    ### train model H ###
    #####################
    from model import model_f_hopf
    model_f = model_f_hopf().to(device)
    model_f.load_state_dict(torch.load(params.model_f_path))
    
    txs = torch.tensor(np.c_[np.ones_like(s1),x1,s1], dtype=torch.float32).to(device)
    numerical = model_f(txs).cpu().detach().numpy()
    ### s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
    eps = (s2 - (s1 + numerical[:,[-1]]*dt_s))/dt_s**.5
    
    bins_x, bins_s, bins_u = params.bins_x, params.bins_s, params.bins_u
    x_inter, s_inter = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    dist, cond_dist, cond_std, cond_mean, idx_used, unif = \
        get_cond_noise(eps, x1, s1, x_inter, s_inter, shape=[bins_x, bins_s, bins_u], n=50)
    
    x1_re = x1[idx_used,:]
    s1_re = s1[idx_used,:]

    x_idx_ = x1_re[:,[0]]>=x_inter
    x_idx = x_idx_.sum(axis=1)-1

    s_idx_ = s1_re[:,[0]]>=s_inter
    s_idx = s_idx_.sum(axis=1)-1
    
    obs = np.c_[np.ones_like(s1_re), x1_re, s1_re]
    std_mean = np.c_[cond_std[x_idx, s_idx], cond_mean[x_idx, s_idx]]
    
    from model import model_std_mean_hopf
    model_H = model_std_mean_hopf().to(device)
    print(f'model H has {count_parameters(model_H)} parameters')

    t0 = time.time()
    model_H = train_model(model_H, inputs=obs, target=std_mean, niter=10000, lr=params.lr)
    print(f'time for training model H: {(time.time()-t0)/60}')

    torch.save(model_H.state_dict(), params.model_std_mean_path)
    
    
    
    
    #####################
    ### train model K ###
    #####################
    from model import model_fs_dist_hopf
    model_K = model_fs_dist_hopf().to(device)
    print(f'model K has {count_parameters(model_K)} parameters')

    x1_re = np.repeat(x1[idx_used,:], bins_u, axis=0)
    s1_re = np.repeat(s1[idx_used,:], bins_u, axis=0)
    unif_re = np.repeat(unif[:,None], idx_used.sum(), axis=1).T.reshape(-1,1)

    obs = np.c_[x1_re, s1_re, unif_re]
    eps = dist[idx_used,:].reshape(-1,1)
    
    t0 = time.time()
    model_K = train_model(model_K, inputs=obs, target=eps, niter=200000, lr=params.lr)
    print(f'time for training model K: {(time.time()-t0)/60}')

    torch.save(model_K.state_dict(), params.model_dist_path)
    
    
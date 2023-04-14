#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:41:43 2022

@author: dliu
"""


import sys 
sys.path.append("..") 
import torch
import time
import numpy as np
from utils import get_inter_grid, get_cond_noise
from misc import train_model, parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    
    data_path = './dump/lorenz/lorenz_data.pth'
    data_dict = torch.load(data_path)
    xs, ts = data_dict['xs'], data_dict['ts']
    
    xs = xs.transpose(1,0)
    xs = xs.numpy()
    
    kx, ks = 1, 2
    assert kx<ks
    x1, x2 = xs[:,:-ks,:-1], xs[:,kx:-ks+kx,:-1]
    s1, s2 = xs[:,:-ks,[-1]], xs[:,ks:,[-1]]
    
    x1, x2 = x1.reshape(-1,2), x2.reshape(-1,2)
    s1, s2 = s1.reshape(-1,1), s2.reshape(-1,1)
    
    dt_x = (ts[kx]-ts[0]).item()
    dt_s = (ts[ks]-ts[0]).item()
    
    diff_x = (x2-x1)/dt_x
    diff_s = (s2-s1)/dt_s
    
    
    
    #####################
    ### train model N ###
    #####################
    from model import model_f_lorenz
    model_f = model_f_lorenz().to(device)
    # model_f.load_state_dict(torch.load('./models/model_N_Lorenz.pt'))
    print(f'model N has {count_parameters(model_f)} parameters')

    inputs = np.c_[np.ones_like(s1), x1, s1]
    target = np.c_[diff_x,diff_s]
    
    t0 = time.time()
    model_f = train_model(model_f, inputs, target, niter=20000, lr=2.5e-4)
    
    print(f'time for training model f: {(time.time()-t0)/60}')

    ### save model
    torch.save(model_f.state_dict(), './models/model_N_Lorenz.pt') 
    
    
    
    #####################
    ### train model H ###
    #####################    
    from model import model_f_lorenz
    model_f = model_f_lorenz().to(device)
    model_f.load_state_dict(torch.load('./models/model_N_Lorenz.pt'))
    
    txs = torch.tensor(np.c_[np.ones_like(s1),x1,s1], dtype=torch.float32).to(device)
    numerical = model_f(txs).cpu().detach().numpy()
    ### s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
    eps = (s2 - (s1 + numerical[:,[-1]]*dt_s))/dt_s**.5
    
    bins_x, bins_s, bins_u = 30, 50, 50
    x_inter, s_inter = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    dist, cond_dist, cond_std, cond_mean, idx_used, unif = \
        get_cond_noise(eps, x1, s1, x_inter, s_inter, shape=[bins_x, bins_s, bins_u], n=30)
    
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
    model_H = train_model(model_H, inputs=obs, target=std_mean, niter=20000, lr=3e-5)
    print(f'time for training model H: {(time.time()-t0)/60}')

    torch.save(model_H.state_dict(), './models/model_H_Lorenz.pt')
    
    
    # model_H(torch.tensor(np.r_[1, xs[0,0,:]],dtype=torch.float32).to(device))
    
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
    model_K = train_model(model_K, inputs=obs, target=eps, niter=250000, lr=5e-4)
    print(f'time for training model K: {(time.time()-t0)/60}')

    torch.save(model_K.state_dict(), './models/model_K_Lorenz.pt')
    
    
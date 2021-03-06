#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:20:04 2022

@author: dliu
"""
import torch
from torch import optim
import numpy as np
import os
import glob

import matplotlib
font = {'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=.6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev = '_cpu' if device.type=='cpu' else ''

class parameters(object):
    def __init__(self, data_type='CGL', n_trainset=1, kx=1, ks=10, x_lower_bound=-float('Inf'), \
                 shape=[30,20,50], batchsize=1024, lr=5e-4):
        assert data_type in ['CGL', 'Saddle_low', 'Saddle_high'], 'wrong data type'
        
        self.n_trainset = n_trainset
        self.x_lower_bound = x_lower_bound
        self.kx = kx
        self.ks = ks
        
        self.bins_x = shape[0]
        self.bins_s = shape[1]
        self.bins_u = shape[2]
        
        self.batchsize = batchsize
        self.lr = lr
        
        self.data_type = data_type
        if data_type=='CGL':
            self.data_pathes = glob.glob(os.path.abspath('.././datasets/Hopf_datasets/*coupled.data'))
        elif data_type=='Saddle_low':
            self.data_pathes = glob.glob(os.path.abspath('.././datasets/Saddle_datasets_low_noise/*data'))
        elif data_type=='Saddle_high':
            self.data_pathes = glob.glob(os.path.abspath('.././datasets/Saddle_datasets_high_noise/*data'))
        self.data_pathes = sorted(self.data_pathes, key=lambda x: int(os.path.split(x)[1].split('_')[0]))

        suffix = '{}_{}_{}_{}{}'.format(self.x_lower_bound,self.ks,self.n_trainset,data_type,dev)
        
        self.model_f_path = os.path.abspath('./models/model_N_{}.pt'.format(suffix))
        self.model_dist_path = os.path.abspath('./models/model_K_{}.pt'.format(suffix))
        self.model_std_mean_path = os.path.abspath('./models/model_H_{}.pt'.format(suffix))
        
        self.model_gen_path = os.path.abspath('./models/gen_{}.pt'.format(suffix))
        self.model_dis_path = os.path.abspath('./models/dis_{}.pt'.format(suffix))



def train_model(model_f, inputs, target, niter, lr, batchsize=1024, fs_mean=None):
    """
    x, sigma is input data
    f_n is ground truth
    flag=True train network N=derivative
    flag=False train network N'=derivative
    """
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    target = torch.tensor(target, dtype=torch.float32).to(device)

    mse = torch.nn.MSELoss() # Mean squared error
    optim_F = optim.Adam(model_f.parameters(), lr=lr)
    
    for i in range(niter):
        idx = np.random.randint(0,inputs.shape[0],batchsize)
        txs = inputs[idx]
        
        ###directly learn ode, N=x'###
        u_out = model_f(txs)
        target_f = target[idx]
        loss_R = torch.mean(mse(u_out, target_f))
        
        if fs_mean is None:
            loss = loss_R
        else:
            loss_physics = torch.mean(mse(u_out[:,-1],fs_mean(txs[:,0],txs[:,1],txs[:,2])))            
            loss = loss_R + 0.5*loss_physics

        loss.backward(retain_graph=True)
        optim_F.step()
        optim_F.zero_grad()
        
        if i%1000==0:
            print('iter:{}. derivative loss: {:.6f}'.format(i, loss.item()))
            
    return model_f




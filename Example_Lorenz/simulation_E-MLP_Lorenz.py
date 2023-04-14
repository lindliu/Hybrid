#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:36:19 2022

@author: dliu
"""


import sys 
sys.path.append("..") 
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import get_dataset
from misc import parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### load model for ODE(x and sigma)
from model import model_f_lorenz
model_f = model_f_lorenz().to(device)
model_f.load_state_dict(torch.load('./models/model_N_Lorenz.pt'))

### load model for std and mean
from model import model_std_mean_hopf
model_H = model_std_mean_hopf().to(device)
model_H.load_state_dict(torch.load('./models/model_H_Lorenz.pt'))

### load model for distribution of noise in sigma
from model import model_fs_dist_hopf
model_K = model_fs_dist_hopf().to(device)
model_K.load_state_dict(torch.load('./models/model_K_Lorenz.pt'))

lr = 5e-4

data_path = './dump/lorenz/lorenz_data.pth'
data_dict = torch.load(data_path)
xs, ts = data_dict['xs'], data_dict['ts']

xs = xs.transpose(1,0)
xs = xs.numpy()

kx, ks = 1, 10
assert kx<ks
x1, x2 = xs[:,:-ks,:-1], xs[:,kx:-ks+kx,:-1]
s1, s2 = xs[:,:-ks,[-1]], xs[:,ks:,[-1]]

x1, x2 = x1.reshape(-1,2), x2.reshape(-1,2)
s1, s2 = s1.reshape(-1,1), s2.reshape(-1,1)

dt_x = (ts[kx]-ts[0]).item()
dt_s = (ts[ks]-ts[0]).item()

diff_x = (x2-x1)/dt_x
diff_s = (s2-s1)/dt_s


def plot(data):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 1)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    z1, z2, z3 = np.split(data, indices_or_sections=3, axis=-1)

    [ax00.plot(z1[i,:,0],z2[i,:,0],z3[i,:,0]) for i in range(data.shape[0])]

    ax00.scatter(z1[:, 0, 0], z2[:, 0, 0], z3[:, 0, 0], marker='x')
    # ax00.set_yticklabels([])
    # ax00.set_xticklabels([])
    # ax00.set_zticklabels([])
    # ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    # ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    # ax00.set_title('Data', fontsize=20)
    # xlim = ax00.get_xlim()
    # ylim = ax00.get_ylim()
    # zlim = ax00.get_zlim()
    
if __name__=="__main__":

    def simulation_with_given_sigma(x_init, s_init, s_real):        
        x_series = [x_init.flatten()]
        s_series = [s_init.flatten()]
        t_series = [np.array([0])]
        fx_series = []
    
        for i in range(length):
            txs = torch.tensor([np.r_[np.array([1]),x_series[-1],s_series[-1]]], dtype=torch.float32).to(device)
            f_numerical = model_f(txs).cpu().detach().numpy()
            
            fx_numerical = f_numerical[:,:-1]
            xx_ = x_series[-1] + fx_numerical*dt ##Euler method
            x_series.append(xx_.flatten())
            
            ss_ = np.array([s_real[i]])   ###using real ss_
            s_series.append(ss_)
            
            t_series.append(t_series[-1]+dt)
            fx_series.append(fx_numerical.flatten())
            
        x_series = np.array(x_series)[1:]
        s_series = np.array(s_series)[1:]
        t_series = np.array(t_series)[1:]
        fx_series = np.array(fx_series)
        return x_series, s_series
    
    length = 90
    dt = 0.020202020183205605
    
    idx = np.random.choice(np.arange(xs.shape[0]), 10)
    init = xs[idx,0,:]
    data_a = []
    for i in range(10):
        x_series, s_series = simulation_with_given_sigma(x_init=init[i,:-1], \
                                                         s_init=init[i,-1], s_real=xs[idx[i],:,-1])
        data = np.c_[x_series, s_series]
        data_a.append(data)
    
    data_a = np.array(data_a)
    plot(data_a)
    
    # idx = np.array([0])
    plt.figure()
    plt.plot(xs[idx[0],:-10,:-1])
    plt.plot(data_a[0,:,:-1])
    print('MSE: {}'.format(np.mean((xs[idx, :-10, :-1]-data_a[:,:,:-1])**2))) # 0.073
    
    
    #################################################################
    ###### simulate both x and s by E-MLP with Gaussian noise #######
    #################################################################
    from simulate_tools import simulate_xs_with_GNN
    
    length = 90
    dt = 0.020202020183205605
    
    idx = np.random.choice(np.arange(xs.shape[0]), 10)
    # idx = np.array([0])
    init = xs[idx,0,:]
    data_a = []
    for i in range(idx.shape[0]):
        x_init, s_init = [init[i,:-1]], [init[i,[-1]]]
        x_series, s_series, t_series = simulate_xs_with_GNN(model_f, model_H, \
                                                            x_init, s_init, dt, length)
        data = np.c_[x_series, s_series]
        data_a.append(data)
        
    data_a = np.array(data_a)
    plot(data_a)


    ##################################################################
    ###### simulate both x and s by E-MLP with empirical noise #######
    ##################################################################
    from simulate_tools import simulate_xs_with_EN
    length = 90
    dt = 0.020202020183205605
    
    idx = np.random.choice(np.arange(xs.shape[0]), 10)
    # idx = np.array([0])
    init = xs[idx,0,:]
    data_a = []
    for i in range(idx.shape[0]):
        x_init, s_init = [init[i,:-1]], [init[i,[-1]]]
        x_series, s_series, t_series = simulate_xs_with_EN(model_f, model_K, \
                                                           x_init, s_init, dt, length)
        data = np.c_[x_series, s_series]
        data_a.append(data)

    data_a = np.array(data_a)
    plot(data_a)
    








    ####################
    ### MMD distance ###
    ####################
    from misc import mmd ## is mmd stable??

    ### E-MLP with Gaussian noise ###
    from simulate_tools import simulate_xs_with_GNN

    length = 100
    dt = 0.020202020183205605

    num = 1024
    idx = np.random.choice(np.arange(xs.shape[0]), num)
    # idx = np.arange(xs.shape[0])
    init = xs[idx,0,:]
    
    X_real = xs[:, :length, :]

    MMD_GNN = []
    for _ in range(10):
        X_GNN_generate = []
        for i in range(idx.shape[0]):
            x_series, s_series, t_series = simulate_xs_with_GNN(model_f, model_H, \
                                                                [init[i,:-1]], [init[i,[-1]]], dt, length)
            data = np.c_[x_series, s_series]
            X_GNN_generate.append(data)
            
            if i==0:
                print(i)
        
        X_GNN_generate = np.array(X_GNN_generate)
        
        mmd1 = mmd(X_real[:,:,0], X_GNN_generate[:,:,0], sigma=1).item()
        mmd2 = mmd(X_real[:,:,1], X_GNN_generate[:,:,1], sigma=1).item()
        mmd3 = mmd(X_real[:,:,2], X_GNN_generate[:,:,2], sigma=1).item()
        
        MMD_GNN.append([mmd1,mmd2,mmd3])
        
    print('MMD mean of GNN: ', np.array(MMD_GNN).mean(0))  #[0.00146437 0.00123623 0.00115157]
    print('MMD std of GNN: ', np.array(MMD_GNN).std(0))  #[2.38849636e-05 9.90456258e-06 3.55381406e-06]
    
    
    ### E-MLP with empirical noise ###
    from simulate_tools import simulate_xs_with_EN
    
    length = 100
    dt = 0.020202020183205605

    num = 1024
    idx = np.random.choice(np.arange(xs.shape[0]), num)
    init = xs[idx,0,:]
    
    X_real = xs[:, :length, :]

    MMD_EN = []
    for _ in range(10):
        X_EN_generate = []
        for i in range(idx.shape[0]):
            x_series, s_series, t_series = simulate_xs_with_EN(model_f, model_K, \
                                                               [init[i,:-1]], [init[i,[-1]]], dt, length)
            data = np.c_[x_series, s_series]
            X_EN_generate.append(data)
            
            if i==0:
                print(i)
        
        X_EN_generate = np.array(X_EN_generate)
        
        mmd1 = mmd(X_real[:,:,0], X_EN_generate[:,:,0], sigma=1).item()
        mmd2 = mmd(X_real[:,:,1], X_EN_generate[:,:,1], sigma=1).item()
        mmd3 = mmd(X_real[:,:,2], X_EN_generate[:,:,2], sigma=1).item()
        
        MMD_EN.append([mmd1,mmd2,mmd3])
        
    print('MMD mean of EN: ', np.array(MMD_EN).mean(0))  #[0.00149115 0.00125104 0.00116394]
    print('MMD std of EN: ', np.array(MMD_EN).std(0))  #[2.44013013e-05 8.44308071e-06 5.18548702e-06]
    
    
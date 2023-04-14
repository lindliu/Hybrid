#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:11:48 2022

@author: dliu
"""

import sys 
sys.path.append("..") 
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import get_dataset
from misc import parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10, lr=5e-4)

data_test_pathes = params.data_pathes[params.n_trainset:params.n_trainset+1]

### load data ###
x_lower_bound = params.x_lower_bound
kx = params.kx
ks = params.ks
x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
    get_dataset(data_test_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)

### load model for ODE(x and sigma)
from model import model_f_hopf
model_f = model_f_hopf().to(device)
model_f.load_state_dict(torch.load(params.model_f_path))

### load model for std and mean
from model import model_std_mean_hopf
model_H = model_std_mean_hopf().to(device)
model_H.load_state_dict(torch.load(params.model_std_mean_path))

### load model for distribution of noise in sigma
from model import model_fs_dist_hopf
model_K = model_fs_dist_hopf().to(device)
model_K.load_state_dict(torch.load(params.model_dist_path))


def hopf_comparison_plot(x_series, s_series, t_series, idx):
    """
    1st row: real data
    2nd row: numerical derivative of real data
    3rd row: generated data: x_series, s_series, t_series
    """
    ### real data and its' numerical derivative
    t_all = np.cumsum(dt_x[idx:])
    mask = t_all<200

    t_ = t_all[mask]
    x_ = x1[idx:][mask]
    s_ = s1[idx:][mask]

    mask_ = t_series.flatten()<200
    x_series_, s_series_, t_series_ = x_series[mask_,:], s_series[mask_,:], t_series[mask_,:]
    
    fig = plt.figure(figsize=[30,22])
    ax1 = fig.add_subplot(2,3,1)
    ax1.plot(t_, x_[:,0], linewidth=.6, label='Part of $X_1$')
    ax1.plot(t_series_, x_series_[:,0], linewidth=.6, label='Generated $X_1$')
    ax1.set_xlabel('t')
    ax1.set_ylabel('$X_1$')
    # ax1.title.set_text('Part of $X_1$')
    ax1.legend(loc='lower right')
    
    ax2 = fig.add_subplot(2,3,2)
    ax2.plot(t_, x_[:,1], linewidth=.6, label='Part of $X_2$')
    ax2.plot(t_series_, x_series_[:,1], linewidth=.6, label='Generated $X_2$')
    ax2.set_xlabel('t')
    ax2.set_ylabel('$X_2$')
    # ax2.title.set_text('Part of $X_2$')
    ax2.legend(loc='lower right')

    ax3 = fig.add_subplot(2,3,3)
    ax3.plot(t_, s_[:,0], linewidth=.6, label=r'Part of $\bar{\sigma}$')
    ax3.plot(t_series_, s_series_[:,0], linewidth=.6, label=r'Generated $\bar{\sigma}$')
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\bar{\sigma}$')
    # ax3.title.set_text(r'Part of $\bar{\sigma}$')
    ax3.legend(loc='lower right')

    idx = np.random.choice(np.arange(s2.shape[0]),50000)
    vmin = min(diff_x[idx][:,[0]].min(), diff_x[idx][:,[1]].min(), diff_s[idx].min())
    vmax = max(diff_x[idx][:,[0]].max(), diff_x[idx][:,[1]].max(), diff_s[idx].max())
    # idx = np.arange(s2.shape[0])
    ax4 = fig.add_subplot(2,3,4,projection='3d')
    im = ax4.scatter(x1[idx][:,[0]], x1[idx][:,[1]], s1[idx], c=diff_x[idx][:,[0]], s=.1, cmap='plasma', vmin=vmin, vmax=vmax)
    ax4.set_xlabel('$X_1$')
    ax4.set_ylabel('$X_2$')
    ax4.set_zlabel(r'$\bar{\sigma}$')
    # fig.colorbar(im)
    ax4.title.set_text('Numerical derivative of $X_1$ \n with step size {}'.format(kx))
    
    ax5 = fig.add_subplot(2,3,5,projection='3d')
    im = ax5.scatter(x1[idx][:,[0]], x1[idx][:,[1]], s1[idx], c=diff_x[idx][:,[1]], s=.1, cmap='plasma', vmin=vmin, vmax=vmax)
    ax5.set_xlabel('$X_1$')
    ax5.set_ylabel('$X_2$')
    ax5.set_zlabel(r'$\bar{\sigma}$')
    # fig.colorbar(im, ax=[ax1,ax2])
    ax5.title.set_text('Numerical derivative of $X_2$ \n with step size {}'.format(kx))
    
    ax6 = fig.add_subplot(2,3,6,projection='3d')
    im = ax6.scatter(x1[idx][:,[0]], x1[idx][:,[1]], s1[idx], c=diff_s[idx], s=.1, cmap='plasma', vmin=vmin, vmax=vmax)
    ax6.set_xlabel('$X_1$')
    ax6.set_ylabel('$X_2$')
    ax6.set_zlabel(r'$\bar{\sigma}$')
    ax6.title.set_text(r'Numerical derivative of $\bar\sigma$'+'\n with step size {}'.format(ks))
    # fig.colorbar(im, ax=[ax4,ax5,ax6],location='bottom',shrink=.8)
    fig.colorbar(im, ax=[ax4,ax5,ax6],shrink=.8)

    # fig.savefig('./figures/CGL_simulation_E-MLP.png',bbox_inches='tight')
    
if __name__=="__main__":

    #####################################################################
    ##### simulation x with given initial condition and sigma ###########
    #####################################################################
    from simulate_tools import simulate_x_with_sigma
    idx, length = 0, 600
    x_init, s_init = x1[idx], s1[idx]
    x_series, s_series, t_series, fx_series = simulate_x_with_sigma(model_f, x_init, s_init, dt_x, idx, s1, length)
    
    mse1 = np.average((x_series[:, 0]-x1[idx:idx+length, 0])**2)
    mse2 = np.average((x_series[:, 1]-x1[idx:idx+length, 1])**2)
    msef1 = np.average((fx_series[:, 0]-diff_x[idx:idx+length, 0])**2)
    msef2 = np.average((fx_series[:, 1]-diff_x[idx:idx+length, 1])**2)
    
    plt.figure(figsize=[20,15])
    plt.subplot(2,2,1)
    plt.plot(t_series, x_series[:, 0], linewidth=.6, label='$X_1$ generated by E-MLP method')
    plt.plot(t_series, x1[idx:idx+length, 0], linewidth=.6, label='$X_1$ real')
    plt.plot([], [], ' ', label="MSE={:.2e}".format(mse1))
    plt.ylim(-3,1.1)
    plt.xlabel('t')
    plt.legend(loc='lower right')
    
    plt.subplot(2,2,2)
    plt.plot(t_series, fx_series[:, 0], linewidth=.6, label="$X_1'$ generated by MLPs")
    plt.plot(t_series, diff_x[idx:idx+length, 0], linewidth=.6, label="$X_1'$ numerical")
    plt.plot([], [], ' ', label="MSE={:.2e}".format(msef1))
    plt.ylim(-.8,.4)
    plt.xlabel('t')
    plt.legend(loc='lower right')
    
    plt.subplot(2,2,3)
    plt.plot(t_series, x_series[:, 1], linewidth=.6, label='$X_2$ generated by E-MLP method')
    plt.plot(t_series, x1[idx:idx+length, 1], linewidth=.6, label='$X_2$ real')
    plt.plot([], [], ' ', label="MSE={:.2e}".format(mse2))
    plt.ylim(-2.2,1.1)
    plt.xlabel('t')
    plt.legend(loc='lower right')
    
    plt.subplot(2,2,4)
    plt.plot(t_series, fx_series[:, 1], linewidth=.6, label="$X_2'$ generated by MLPs")
    plt.plot(t_series, diff_x[idx:idx+length, 1], linewidth=.6, label="$X_2'$ numerical")
    plt.plot([], [], ' ', label="MSE={:.2e}".format(msef2))
    plt.ylim(-.8,.4)
    plt.xlabel('t')
    plt.legend(loc='lower right')
    
    plt.savefig('./figures/CGL_simulation_with_given_sigma_E-MLP.png',bbox_inches='tight')
    
    ##############################
    ####### estimate E-MLP #######
    ##### for table in paper #####
    ##############################
    length = 600
    np.random.seed(1)
    init_idx = np.random.choice(np.arange(x1.shape[0]-length),1000)
    
    mse1_list, mse2_list, msef1_list, msef2_list = [], [], [], []
    mse_list, msef_list = [], []
    for i, idx in enumerate(init_idx):
        x_init, s_init = x1[idx], s1[idx]
        x_series, s_series, t_series, fx_series = simulate_x_with_sigma(model_f, x_init, s_init, dt_x, idx, s1, length)
        
        mse1 = np.average((x_series[:, 0]-x1[idx:idx+length, 0])**2)
        mse2 = np.average((x_series[:, 1]-x1[idx:idx+length, 1])**2)
        msef1 = np.average((fx_series[:, 0]-diff_x[idx:idx+length, 0])**2)
        msef2 = np.average((fx_series[:, 1]-diff_x[idx:idx+length, 1])**2)
        
        mse1_list.append(mse1)
        mse2_list.append(mse2)
        msef1_list.append(msef1)
        msef2_list.append(msef2)
        
        
        mse_list.append(np.mean((x_series-x1[idx:idx+length, :])**2))
        msef_list.append(np.mean((fx_series-diff_x[idx:idx+length, :])**2))
    
        print(i)

    print("MSE of X1: {}".format(np.mean(mse1_list))) 
    print("MSE of X2: {}".format(np.mean(mse2_list))) 
    print("MSE of X1': {}".format(np.mean(msef1_list))) 
    print("MSE of X2': {}".format(np.mean(msef2_list)))
    
    print("MSE of X: {}".format(np.mean(mse_list)))
    print("MSE of X'': {}".format(np.mean(msef_list)))
    
    
    
    #################################################################
    ###### simulate both x and s by E-MLP with Gaussian noise #######
    #################################################################
    from simulate_tools import simulate_xs_with_GNN
    length = 3500
    idx = 100
    dt = 0.11
    x_init, s_init = [x1[idx]], [s1[idx]]
    x_series, s_series, t_series = simulate_xs_with_GNN(model_f, model_H, x_init, s_init, dt, length)
    
    hopf_comparison_plot(x_series, s_series, t_series, idx)
    
    
    ##################################################################
    ###### simulate both x and s by E-MLP with empirical noise #######
    ##################################################################
    from simulate_tools import simulate_xs_with_EN
    length = 3500
    idx = 100
    dt = 0.11
    
    Hopf_s_list = []
    for _ in range(3):
        x_init, s_init = [x1[idx]], [s1[idx]]
        x_series, s_series, t_series = simulate_xs_with_EN(model_f, model_K, x_init, s_init, dt, length, lamb=0.210)
        
        Hopf_s_list.append(np.c_[t_series, x_series, s_series])
        
    hopf_comparison_plot(x_series, s_series, t_series, idx)
    
    # ########### save data ##############
    # mask_ = t_series.flatten()<120
    # Hopf_s_list = [hopf[mask_, :] for hopf in Hopf_s_list]
    # np.save('./figures/Hopf_simulation.npy', Hopf_s_list)
    
    # t_all = np.cumsum(dt_x[idx:])
    # mask = t_all<120
    # t_ = t_all[mask]
    # x_ = x1[idx:][mask]
    # s_ = s1[idx:][mask]
    # np.save('./figures/Hopf_part.npy', np.c_[t_, x_, s_])
    
    
    
    
    
    
    
    ##################
    ###### MMD #######
    ##################
    from misc import mmd
    ### simulate both x and sigma with Gaussian noise, std and mean by NN
    def simulate_xs_with_GNN_irregular(model_f, model_std_mean, x_init, s_init, dt, length, stop_threshold=-20):
        
        x_series, s_series, t_series = [x_init], [s_init], [np.array([0])]
        for i in range(length-1):
            txs = torch.tensor(np.array([np.r_[np.array([1]),x_series[-1],s_series[-1]]]), dtype=torch.float32).to(device)
            f_numerical = model_f(txs).cpu().detach().numpy()
            
            fx_numerical = f_numerical[:,:-1]
            xx_ = x_series[-1] + fx_numerical*dt[i] ##Euler method
            x_series.append(xx_.flatten())
            
            std_mean = model_std_mean(txs).cpu().detach().numpy()
            std, mean = std_mean[:,0], std_mean[:,1]
            
            fs_numerical = f_numerical[:,[-1]] 
            ss_ = s_series[-1] + fs_numerical*dt[i] + (np.random.randn()*std+mean)*dt[i]**.5 ##Euler method, consider noise is Brownian motion
            s_series.append(ss_.flatten())
            
            t_series.append(t_series[-1]+dt[i])
            
            if xx_.flatten()[0]<stop_threshold: ##for saddle data
                break
        x_series, s_series, t_series = np.array(x_series), np.array(s_series), np.array(t_series)
        return x_series, s_series, t_series
    
    data_test_pathes = params.data_pathes[params.n_trainset:params.n_trainset+1]
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_test_pathes, x_lower_bound=-100, kx=1, ks=1, resample=False)
    
    
    length = 256
    num = 100
    # init_ = np.random.choice(np.arange(91200-600),1024)
    mm1 = np.logical_and(x1[:,0]>-0.05, x1[:,0]<0.05)
    mm2 = np.logical_and(x1[:,1]>-1.05, x1[:,1]<-0.95)
    mm = np.logical_and(mm1, mm2)
    mm[-length:] = False
    init_ = np.random.choice(np.arange(x1.shape[0])[mm], num)
    
    
    x1_train, s1_train = [], []
    for i in range(length):
        x1_train.append(x1[init_+i])
        s1_train.append(s1[init_+i])
    
    x1_train = np.array(x1_train, dtype=np.float32)
    s1_train = np.array(s1_train, dtype=np.float32)
    X_real = np.c_[x1_train,s1_train].transpose(1,0,2)
    

    MMD_GNN = []
    for _ in range(10):
        X_GNN_generate = []
        for l in range(num):
            idx = init_[l]
            dt = dt_x[idx:,0]
            x_init, s_init = x1[idx], s1[idx]
            
            x_series, s_series, t_series = simulate_xs_with_GNN_irregular(model_f, model_H, x_init, s_init, dt, length)
            X_GNN_generate.append(np.c_[x_series, s_series])
            
            if l%100==0:
                print(l)
        
        X_GNN_generate = np.array(X_GNN_generate)
        
        mmd1 = mmd(X_real[:,:,0], X_GNN_generate[:,:,0], sigma=1).item()
        mmd2 = mmd(X_real[:,:,1], X_GNN_generate[:,:,1], sigma=1).item()
        mmd3 = mmd(X_real[:,:,2], X_GNN_generate[:,:,2], sigma=1).item()
        
        MMD_GNN.append([mmd1,mmd2,mmd3])
        
    print('MMD mean of GNN: ', np.array(MMD_GNN).mean(0))
    print('MMD std of GNN: ', np.array(MMD_GNN).std(0))
    
    
    # plt.plot(x_series[:,:])
    # plt.plot(x1[idx:idx+length,:])
    
    
    
    
    ### simulate both x and sigma with approximated empirical noise(model_eps)
    def simulate_xs_with_EN_irregular(model_f, model_eps, x_init, s_init, dt, length=10000, stop_threshold=-20, lamb=0.210):
        x_series, s_series, t_series = [x_init], [s_init], [np.array([0])]
         # dt is larger than dt in given data
        for i in range(length-1):
            txs = torch.tensor([np.r_[np.array([1]),x_series[-1],s_series[-1]]], dtype=torch.float32).to(device)
            f_numerical = model_f(txs).cpu().detach().numpy()
            
            fx_numerical = f_numerical[:,:-1]
            xx_ = x_series[-1] + fx_numerical*dt[i] ##Euler method
            x_series.append(xx_.flatten())
        
            obs = torch.tensor([np.r_[x_series[-1],s_series[-1], np.random.rand()]], dtype=torch.float32).to(device)
            eps = model_eps(obs).cpu().detach().numpy().item()
            
            fs_numerical = f_numerical[:,[-1]]
            # fs_numerical = fs_np(1,xx_,ss_)
            ss_ = s_series[-1] + fs_numerical*dt[i] + eps*dt[i]**lamb ##Euler method
            s_series.append(ss_.flatten())
            
            t_series.append(t_series[-1]+dt[i])
                
            if xx_.flatten()[0]<stop_threshold: ##for saddle data
                break
            
        x_series = np.array(x_series)
        s_series = np.array(s_series)
        t_series = np.array(t_series)
        return x_series, s_series, t_series
        
    
    MMD_EN = []
    for _ in range(10):
        X_EN_generate = []
        for l in range(num):
            # idx = np.random.choice(init_)
            idx = init_[l]
            dt = dt_x[idx:,0]
            x_init, s_init = x1[idx], s1[idx]
            
            x_series, s_series, t_series = simulate_xs_with_EN_irregular(model_f, model_K, x_init, s_init, dt, length, lamb=.21)
            X_EN_generate.append(np.c_[x_series, s_series])
            
            if l%100==0:
                print(l)
    
        X_EN_generate = np.array(X_EN_generate)
        
        mmd1 = mmd(X_real[:,:,0], X_EN_generate[:,:,0], sigma=1).item()
        mmd2 = mmd(X_real[:,:,1], X_EN_generate[:,:,1], sigma=1).item()
        mmd3 = mmd(X_real[:,:,2], X_EN_generate[:,:,2], sigma=1).item()
        
        MMD_EN.append([mmd1,mmd2,mmd3])
        

    print('MMD mean of EN: ', np.array(MMD_EN).mean(0))
    print('MMD std of EN: ', np.array(MMD_EN).std(0))
    
    
    
    X_NeuralSDE = np.load('./figures/NeuralSDE/samples.npy')
    print('MMD of X1 of Neural SDE: ', mmd(X_real[:,:,0], X_NeuralSDE[:,:,0], sigma=1))
    print('MMD of X2 of Neural SDE: ', mmd(X_real[:,:,1], X_NeuralSDE[:,:,1], sigma=1))
    print('MMD of sigma bar of Neural SDE: ', mmd(X_real[:,:,2], X_NeuralSDE[:,:,2], sigma=1))
    
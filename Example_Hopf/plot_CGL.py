#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:07:24 2022

@author: dliu
"""


import sys 
sys.path.append("..") 
import torch
import numpy as np
from utils import get_dataset, get_inter_grid, get_cond_noise
from misc import parameters
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10, lr=5e-4)


def plot_result_H(obs, std_mean, targ_out):
    fig, axes = plt.subplots(2,3,figsize=[15,7])
    axes = axes.flatten()
    
    vmin = min(targ_out[:,0].min(), std_mean[:,0].min())
    vmax = max(targ_out[:,0].max(), std_mean[:,0].max())
    im = axes[0].scatter(obs[:,1], obs[:,3], c=targ_out[:,0], s=.1, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('$x_0$')
    axes[0].set_ylabel(r'$\bar{\sigma}$')
    axes[0].title.set_text('predict std')
    
    im = axes[1].scatter(obs[:,1], obs[:,3], c=std_mean[:,0], s=.1, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('$x_0$')
    axes[1].set_ylabel(r'$\bar{\sigma}$')
    axes[1].title.set_text('statistical std')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[:2], shrink=0.95)

    im = axes[2].scatter(obs[:,1], obs[:,3], c=np.abs(targ_out[:,0]-std_mean[:,0]), s=.1)
    axes[2].set_xlabel('$x_0$')
    axes[2].set_ylabel(r'$\bar{\sigma}$')
    axes[2].title.set_text('absolute error')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[2], shrink=0.95)

    vmin = min(targ_out[:,1].min(), std_mean[:,1].min())
    vmax = max(targ_out[:,1].max(), std_mean[:,1].max())
    im = axes[3].scatter(obs[:,1], obs[:,3], c=targ_out[:,1], s=.1, vmin=vmin, vmax=vmax)
    axes[3].set_xlabel('$x_0$')
    axes[3].set_ylabel(r'$\bar{\sigma}$')
    axes[3].title.set_text('predict mean')
    
    im = axes[4].scatter(obs[:,1], obs[:,3], c=std_mean[:,1], s=.1, vmin=vmin, vmax=vmax)
    axes[4].set_xlabel('$x_0$')
    axes[4].set_ylabel(r'$\bar{\sigma}$')
    axes[4].title.set_text('statistical mean')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[3:5], shrink=0.95)

    im = axes[5].scatter(obs[:,1], obs[:,3], c=np.abs(targ_out[:,1]-std_mean[:,1]), s=.1)
    axes[5].set_xlabel('$x_0$')
    axes[5].set_ylabel(r'$\bar{\sigma}$')
    axes[5].title.set_text('absolute error')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[5], shrink=0.95)
    
    fig.suptitle('Prediction of model H')
    fig.savefig('./figures/CGL_H_std_mean.png',bbox_inches='tight')


def plot_result_K(obs, eps, eps_out):

    fig = plt.figure(figsize=[30,12])
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.scatter(obs[:,0], obs[:,2], obs[:,3], c=eps_out, s=.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel(r'$\bar{\sigma}$')
    ax.set_zlabel('distribution')
    ax.set_title('estimated empirical distribution')
    
    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.scatter(obs[:,0], obs[:,2], obs[:,3], c=eps, s=.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel(r'$\bar{\sigma}$')
    ax.set_zlabel('distribution')
    ax.set_title('empirical distribution')
    
    ax = fig.add_subplot(1,3,3)
    k = 1
    ax.plot(eps_out.flatten()[k*bins_u:(k+1)*bins_u], unif, label='estimated distribution')
    ax.plot(eps.flatten()[k*bins_u:(k+1)*bins_u], unif, label='empirical distribution')
    ax.legend()
    ax.set_title('Example distribution')
    ax.set_aspect(.03)
    fig.savefig('./figures/CGL_K_distribution.png',bbox_inches='tight')

if __name__=="__main__":

    data_train_pathes = params.data_pathes[:params.n_trainset]
    ### load data ###
    x_lower_bound = params.x_lower_bound
    kx = params.kx
    ks = params.ks
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)


    ###########################
    ### plot model H result ###
    ###########################
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

    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    
    
    from model import model_std_mean_hopf
    model_H = model_std_mean_hopf().to(device)
    ##### compare estimated distribution by NN with empirical distribution
    model_H.load_state_dict(torch.load(params.model_std_mean_path))
    
    targ_out = model_H(obs_tensor).cpu().detach().numpy()
    
    plot_result_H(obs, std_mean, targ_out)
    
    
    
    
    
    
    ###########################
    ### plot model K result ###
    ###########################
    from model import model_fs_dist_hopf

    ##### compare estimated distribution by NN with empirical distribution
    model_K = model_fs_dist_hopf().to(device)
    model_K.load_state_dict(torch.load(params.model_dist_path))
    
    x1_ = x1[idx_used,:]
    s1_ = s1[idx_used,:]
    target_ = dist[idx_used,:]
    
    idx = np.random.randint(0,x1_.shape[0],1000)
    x1_re = np.repeat(x1_[idx,:], bins_u, axis=0)
    s1_re = np.repeat(s1_[idx,:], bins_u, axis=0)
    unif_re = np.repeat(unif[:,None], idx.shape[0], axis=1).T.reshape(-1,1)

    obs = np.c_[x1_re, s1_re, unif_re]
    eps = target_[idx,:].reshape(-1,1)

    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    eps_tensor = torch.tensor(eps, dtype=torch.float32).to(device) 
    
    eps_out = model_K(obs_tensor).cpu().detach().numpy()
    
    plot_result_K(obs, eps, eps_out)
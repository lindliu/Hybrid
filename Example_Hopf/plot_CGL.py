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
    # fig.savefig('./figures/CGL_H_std_mean.eps', dpi=300, bbox_inches='tight')


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
    # fig.savefig('./figures/CGL_K_distribution.eps', dpi=300, bbox_inches='tight')



def compare_distribution_rare_ordinary_case(obs, eps, eps_out, model_H, cond_dist_test, x_inter, s_inter):
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1.4])
    axsLeft = subfigs[0].subplots(1, 1, sharey=True)
    axsRight = subfigs[1].subplots(2, 1, sharex=True)
    
    k_r = np.argmin(obs[:,0])//50  #distribution at rare event
    case_rare = obs[[k_r*50],:-1]
    mm=np.logical_and(obs[:,0]>0.,obs[:,0]<0.1)
    k_c = np.random.choice(np.arange(obs.shape[0])[mm])//50
    case_ordinary = obs[[k_c*50],:-1]
    
    axsLeft.scatter(x1_[:,0],s1_, s=.1)
    axsLeft.scatter(case_rare[0,0], case_rare[0,-1], marker="o", s=100, c='r')
    # axsLeft.annotate('rare case',  xy=(case_rare[0,0], case_rare[0,-1]))
    axsLeft.text(case_rare[0,0], case_rare[0,-1], s='case 1', fontsize='xx-large')
    axsLeft.scatter(case_ordinary[0,0], case_ordinary[0,-1], marker="o", s=100, c='r')
    # axsLeft.annotate('ordinary case',  xy=(case_ordinary[0,0], case_ordinary[0,-1]))
    axsLeft.text(case_ordinary[0,0], case_ordinary[0,-1], s='case 2', fontsize='xx-large')
    axsLeft.set_title('CGL training datasets scatter plot', fontsize=25)
    axsLeft.set_xlabel('$x_1$', fontsize=25)
    axsLeft.set_ylabel(r'$\bar{\sigma}$', fontsize=25)
    
    ##############################
    ### rare case distribution ###
    ##############################
    ### distribution by empirical noise
    axsRight[0].plot(eps_out.flatten()[k_r*bins_u:(k_r+1)*bins_u], unif, label='Predicted Empirical Distribution')
    
    ### distribution by Gaussian noise
    conditions = torch.tensor(np.c_[np.array([1]), case_rare], dtype=torch.float32).to(device)
    std_mean = model_H(conditions).cpu().detach().numpy()
    from scipy.stats import norm
    std, mean = std_mean[0,0], std_mean[0,1]
    x = np.linspace(eps.flatten()[k_r*bins_u:(k_r+1)*bins_u].min(), eps.flatten()[k_r*bins_u:(k_r+1)*bins_u].max(), 100)
    axsRight[0].plot(x, norm.cdf(x, loc=mean, scale=std), label='Predicted Gaussian Distribution')
    
    ### empirical distribution from train dataset
    axsRight[0].plot(eps.flatten()[k_r*bins_u:(k_r+1)*bins_u], unif, label='Empirical Distribution from trainsets')
    
    ### empirical distribution from test dataset
    i = (case_rare[0,0]>x_inter).sum()-1
    j = (case_rare[0,-1]>s_inter).sum()-1
    axsRight[0].plot(cond_dist_test[i,j,:], unif, label='Empirical Distribution from testsets')
    
    axsRight[0].legend(loc='upper left', fontsize=20)
    axsRight[0].set_title('The distribution of noise for "case 1"', fontsize=25)
    
    ##################################
    ### ordinary case distribution ###
    ##################################
    ### distribution by empirical noise
    axsRight[1].plot(eps_out.flatten()[k_c*bins_u:(k_c+1)*bins_u], unif, label='Predicted Empirical Distribution')
    
    ### distribution by Gaussian noise
    conditions = torch.tensor(np.c_[np.array([1]), case_ordinary], dtype=torch.float32).to(device)
    std_mean = model_H(conditions).cpu().detach().numpy()
    from scipy.stats import norm
    std, mean = std_mean[0,0], std_mean[0,1]
    x = np.linspace(eps.flatten()[k_c*bins_u:(k_c+1)*bins_u].min(), eps.flatten()[k_c*bins_u:(k_c+1)*bins_u].max(), 100)
    axsRight[1].plot(x, norm.cdf(x, loc=mean, scale=std), label='Predicted Gaussian Distribution')
    
    ### empirical distribution from train dataset
    axsRight[1].plot(eps.flatten()[k_c*bins_u:(k_c+1)*bins_u], unif, label='Empirical Distribution from trainsets')
    
    ### empirical distribution from test dataset
    i = (case_ordinary[0,0]>x_inter).sum()-1
    j = (case_ordinary[0,-1]>s_inter).sum()-1
    axsRight[1].plot(cond_dist_test[i,j,:], unif, label='Empirical Distribution from testsets')
    axsRight[1].set_xlim(-0.038,0.025)
    axsRight[1].legend(loc='upper left', fontsize=20)
    axsRight[1].set_title('The distribution of noise for "case 2"', fontsize=25)
    
    # fig.savefig('./figures/CGL_rare_ordinary_case.eps', dpi=300,  bbox_inches='tight')




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

    obs_K = np.c_[x1_re, s1_re, unif_re]
    eps = target_[idx,:].reshape(-1,1)

    obs_tensor = torch.tensor(obs_K, dtype=torch.float32).to(device)
    eps_tensor = torch.tensor(eps, dtype=torch.float32).to(device) 
    
    eps_out = model_K(obs_tensor).cpu().detach().numpy()
    
    plot_result_K(obs_K, eps, eps_out)
    
    
    
    
    
    


    ###################################################################################
    ### compare distribution(train,test,estimated empirical and estimated Gaussian) ###
    ###                     in rare and ordinary case                               ###
    ###################################################################################
    data_test_pathes = params.data_pathes[params.n_trainset:params.n_trainset+1]
    x1_test, x2_test, s1_test, s2_test, diff_x_test, diff_s_test, dt_x_test, dt_s_test = \
        get_dataset(data_test_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)


    txs = torch.tensor(np.c_[np.ones_like(s1_test),x1_test,s1_test], dtype=torch.float32).to(device)
    numerical = model_f(txs).cpu().detach().numpy()
    ### s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
    eps_test = (s2_test - (s1_test + numerical[:,[-1]]*dt_s_test))/dt_s_test**.5
    
    bins_x, bins_s, bins_u = params.bins_x, params.bins_s, params.bins_u
    x_inter_, s_inter_ = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    dist_test, cond_dist_test, cond_std_test, cond_mean_test, idx_used_test, unif_test = \
        get_cond_noise(eps_test, x1_test, s1_test, x_inter_, s_inter_, shape=[bins_x, bins_s, bins_u], n=100)
    
    
    compare_distribution_rare_ordinary_case(obs_K, eps, eps_out, model_H, cond_dist_test, x_inter_, s_inter_)
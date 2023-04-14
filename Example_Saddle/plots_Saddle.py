#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:43:28 2022

@author: dliu
"""

import sys 
sys.path.append("..") 
import numpy as np
import torch
from utils import fx, fs_saddle_high, get_dataset, get_cond_noise, get_inter_grid
from misc import parameters
import matplotlib.pyplot as plt
plt.set_cmap('plasma')

import matplotlib
font = {'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=.6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_result_N(model_f, x1, s1, diff_x, diff_s):
    # plt.set_cmap('plasma')

    x_len, s_len = 71, 51
    x_ = np.linspace(params.x_lower_bound, 1, x_len)
    s_ = np.linspace(0.4, 0.6, s_len)
    xx_, ss_ = np.meshgrid(x_, s_, indexing="ij")
    xx = torch.tensor(xx_.flatten()[:,None], dtype=torch.float32).to(device)
    ss = torch.tensor(ss_.flatten()[:,None], dtype=torch.float32).to(device)
    
    fig, axes = plt.subplots(2,4,figsize=[25,10])
    fig.tight_layout()
    # plt.subplots_adjust(hspace=0)
    axes = axes.flatten()
    
    f_real = fx
    real = f_real(1,xx,ss).reshape(xx_.shape)
    real = real.cpu().numpy()
    
    txs = torch.cat((torch.ones_like(xx), xx,ss),1)
    numerical = model_f(txs)
    numerical = numerical[:,0].reshape(xx_.shape).cpu().detach().numpy()
    
    vmin = min(real.min(), numerical.min(), diff_x.min())
    vmax = max(real.max(), numerical.max(), diff_x.max())
    
    im = axes[0].scatter(x1, s1, c=diff_x, s=.05, vmin=vmin, vmax=vmax)
    axes[0].set_yticks([0.4, 0.5, 0.6])
    axes[0].set_yticklabels([0.4, 0.5, 0.6])
    axes[0].set_xticks([-1,0,1])
    axes[0].set_xticklabels([-1,0,1])
    axes[0].set_ylabel(r'$\bar{\sigma}$')
    axes[0].set_xlabel('X')
    axes[0].title.set_text("numerical X' data")
    axes[0].set_aspect(8)
    
    im = axes[1].imshow(numerical.T[::-1], vmin=vmin, vmax=vmax)
    axes[1].set_yticks([0,s_len//2,s_len-1])
    axes[1].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[1].set_xticks([0,x_len//2,x_len-1])
    axes[1].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[1].set_ylabel(r'$\bar{\sigma}$')
    axes[1].set_xlabel('X')
    axes[1].title.set_text("Predicted X'")
    
    im = axes[2].imshow(real.T[::-1], vmin=vmin, vmax=vmax)
    axes[2].set_yticks([0,s_len//2,s_len-1])
    axes[2].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[2].set_xticks([0,x_len//2,x_len-1])
    axes[2].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[2].set_ylabel(r'$\bar{\sigma}$')
    axes[2].set_xlabel('X')
    axes[2].title.set_text("Real X'")
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[:3], shrink=0.7)
    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['low', 'medium', 'high'])

    abs_error_x = np.abs(real-numerical)
    im = axes[3].imshow(abs_error_x.T[::-1])
    axes[3].set_yticks([0,s_len//2,s_len-1])
    axes[3].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[3].set_xticks([0,x_len//2,x_len-1])
    axes[3].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[3].set_ylabel(r'$\bar{\sigma}$')
    axes[3].set_xlabel('X')
    axes[3].title.set_text("absolute error")
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[3], shrink=0.7)
    
    
    f_real = fs_saddle_high
    real = f_real(1,xx,ss).reshape(xx_.shape)
    real = real.cpu().numpy()
    
    txs = torch.cat((torch.ones_like(xx), xx,ss),1)
    numerical = model_f(txs)
    numerical = numerical[:,1].reshape(xx_.shape).cpu().detach().numpy()
    
    vmin = min(real.min(), numerical.min(), diff_s.min())
    vmax = max(real.max(), numerical.max(), diff_s.max())
    
    im = axes[4].scatter(x1, s1, c=diff_s, s=.05, vmin=vmin, vmax=vmax)
    axes[4].set_yticks([0.4, 0.5, 0.6])
    axes[4].set_yticklabels([0.4, 0.5, 0.6])
    axes[4].set_xticks([-1,0,1])
    axes[4].set_xticklabels([-1,0,1])
    axes[4].set_ylabel(r'$\bar{\sigma}$')
    axes[4].set_xlabel('X')
    axes[4].title.set_text(r"numerical $\bar{\sigma}$' data")
    axes[4].set_aspect(8)

    im1 = axes[5].imshow(numerical.T[::-1], vmin=vmin, vmax=vmax)
    axes[5].set_yticks([0,s_len//2,s_len-1])
    axes[5].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[5].set_xticks([0,x_len//2,x_len-1])
    axes[5].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[5].set_ylabel(r'$\bar{\sigma}$')
    axes[5].set_xlabel('X')
    axes[5].title.set_text(r"Predicted $\bar{\sigma}$'")
    
    im1 = axes[6].imshow(real.T[::-1], vmin=vmin, vmax=vmax)
    axes[6].set_yticks([0,s_len//2,s_len-1])
    axes[6].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[6].set_xticks([0,x_len//2,x_len-1])
    axes[6].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[6].set_ylabel(r'$\bar{\sigma}$')
    axes[6].set_xlabel('X')
    axes[6].title.set_text(r"Mean field $\bar{\sigma}$'")
    

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist()[4:7], shrink=0.7)
    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['low', 'medium', 'high'])

    abs_error_sigma = np.abs(real-numerical)
    im1 = axes[7].imshow(abs_error_sigma.T[::-1])
    axes[7].set_yticks([0,s_len//2,s_len-1])
    axes[7].set_yticklabels(np.round(s_[::-1][[0,s_len//2,s_len-1]],2))
    axes[7].set_xticks([0,x_len//2,x_len-1])
    axes[7].set_xticklabels(np.round(x_[[0,x_len//2,x_len-1]],2))
    axes[7].set_ylabel(r'$\bar{\sigma}$')
    axes[7].set_xlabel('X')
    axes[7].title.set_text("absolute error")
    
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist()[7], shrink=0.7)
    
    # fig.savefig('./figures/N_Saddle.eps', dpi=300, bbox_inches='tight')
    
    print("The mean abs error for x is: {:.4f} and the max abs error is: {:.4f}".format(np.average(abs_error_x), abs_error_x.max()))
    print("The mean abs error for sigma is: {:.4f} and the max abs error is: {:.4f}".format(np.average(abs_error_sigma), abs_error_sigma.max()))


def plot_result_H(obs, targ_out, std_mean):
    fig, axes = plt.subplots(2,3,figsize=[15,7])
    axes = axes.flatten()
    
    vmin = min(targ_out[:,0].min(), std_mean[:,0].min())
    vmax = max(targ_out[:,0].max(), std_mean[:,0].max())
    im = axes[0].scatter(obs[:,1], obs[:,2], c=targ_out[:,0], s=.5, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('$x_0$')
    axes[0].set_ylabel(r'$\bar{\sigma}$')
    axes[0].title.set_text('predict std')
    
    im = axes[1].scatter(obs[:,1], obs[:,2], c=std_mean[:,0], s=.5, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('$x_0$')
    axes[1].set_ylabel(r'$\bar{\sigma}$')
    axes[1].title.set_text('real std')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[:2], shrink=0.95)

    im = axes[2].scatter(obs[:,1], obs[:,2], c=np.abs(targ_out[:,0]-std_mean[:,0]), s=.1)
    axes[2].set_xlabel('$x_0$')
    axes[2].set_ylabel(r'$\bar{\sigma}$')
    axes[2].title.set_text('absolute error')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[2], shrink=0.95)

    vmin = min(targ_out[:,1].min(), std_mean[:,1].min())
    vmax = max(targ_out[:,1].max(), std_mean[:,1].max())
    im = axes[3].scatter(obs[:,1], obs[:,2], c=targ_out[:,1], s=.5, vmin=vmin, vmax=vmax)
    axes[3].set_xlabel('$x_0$')
    axes[3].set_ylabel(r'$\bar{\sigma}$')
    axes[3].title.set_text('predict mean')
    
    im = axes[4].scatter(obs[:,1], obs[:,2], c=std_mean[:,1], s=.5, vmin=vmin, vmax=vmax)
    axes[4].set_xlabel('$x_0$')
    axes[4].set_ylabel(r'$\bar{\sigma}$')
    axes[4].title.set_text('real mean')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[3:5], shrink=0.95)

    im = axes[5].scatter(obs[:,1], obs[:,2], c=np.abs(targ_out[:,1]-std_mean[:,1]), s=.1)
    axes[5].set_xlabel('$x_0$')
    axes[5].set_ylabel(r'$\bar{\sigma}$')
    axes[5].title.set_text('absolute error')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist()[5], shrink=0.95)
    
    # fig.savefig('./figures/H_Saddle.eps', dpi=300, bbox_inches='tight')


def plot_result_K_H(obs, eps, eps_out, model_H):
    fig = plt.figure(figsize=[30,10])
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.scatter(obs[:,0], obs[:,1], obs[:,2], c=eps_out, s=.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel(r'$\bar{\sigma}$')
    ax.set_zlabel('distribution')
    ax.set_title('estimated distribution')
    
    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.scatter(obs[:,0], obs[:,1], obs[:,2], c=eps, s=.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel(r'$\bar{\sigma}$')
    ax.set_zlabel('distribution')
    ax.set_title('empirical distribution')
    
    ax = fig.add_subplot(1,3,3)
    k = np.argmin(obs[:,0])//50  #distribution at rare event
    mm=np.logical_and(obs[:,0]>0.2,obs[:,0]<0.3)
    k = np.random.choice(np.arange(obs.shape[0])[mm])//50
    ### distribution by empirical noise
    ax.plot(eps_out.flatten()[k*bins_u:(k+1)*bins_u], unif, label='Estimated Empirical Distribution')
    
    ### distribution by Gaussian noise
    conditions = torch.tensor(np.c_[np.array([1]), obs[[k],:-1]], dtype=torch.float32).to(device)
    std_mean = model_H(conditions).cpu().detach().numpy()
    from scipy.stats import norm
    std, mean = std_mean[0,0], std_mean[0,1]
    x = np.linspace(eps.flatten()[k*bins_u:(k+1)*bins_u].min(), eps.flatten()[k*bins_u:(k+1)*bins_u].max(), 100)
    ax.plot(x, norm.cdf(x, loc=mean, scale=std), label='Estimated Gaussian Distribution')
    
    ### empirical distribution from test dataset
    ax.plot(eps.flatten()[k*bins_u:(k+1)*bins_u], unif, label='Test dataset Empirical Distribution')

    ax.legend()
    ax.set_title('example distribution')
    
    # fig.savefig('./figures/K_Saddle.eps', dpi=300, bbox_inches='tight')
    


def compare_distribution_rare_ordinary_case(obs, eps, eps_out, model_H, cond_dist_test, x_inter, s_inter):
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1.4])
    axsLeft = subfigs[0].subplots(1, 1, sharey=True)
    axsRight = subfigs[1].subplots(2, 1, sharex=True)
    
    k_r = np.argmin(obs[:,0])//50  #distribution at rare event
    case_rare = obs[[k_r*50],:-1]
    mm=np.logical_and(obs[:,0]>0.2,obs[:,0]<0.3)
    k_c = np.random.choice(np.arange(obs.shape[0])[mm])//50
    case_ordinary = obs[[k_c*50],:-1]
    
    axsLeft.scatter(x1_,s1_, s=.1)
    axsLeft.scatter(case_rare[0,0], case_rare[0,-1], marker="o", s=100, c='r')
    # axsLeft.annotate('rare case',  xy=(case_rare[0,0], case_rare[0,-1]))
    axsLeft.text(case_rare[0,0], case_rare[0,-1], s='rare event', fontsize='xx-large')
    axsLeft.scatter(case_ordinary[0,0], case_ordinary[0,-1], marker="o", s=100, c='r')
    # axsLeft.annotate('ordinary case',  xy=(case_ordinary[0,0], case_ordinary[0,-1]))
    axsLeft.text(case_ordinary[0,0], case_ordinary[0,-1], s='ordinary event', fontsize='xx-large')
    axsLeft.set_title('Saddle training datasets scatter plot', fontsize=25)
    axsLeft.set_xlabel('x', fontsize=25)
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
    axsRight[0].set_title('The distribution of noise for "rare event"', fontsize=25)
    
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
    axsRight[1].legend(loc='upper left', fontsize=20)
    axsRight[1].set_title('The distribution of noise for "ordinary event"', fontsize=25)
    
    # fig.savefig('./figures/Saddle_rare_ordinary_case.png', dpi=300, bbox_inches='tight')


    
if __name__=="__main__":
    
    ###############################
    ##### Figure 1 ################
    ###############################
    params = parameters(data_type='Saddle_high', n_trainset=20, kx=1, ks=10, x_lower_bound=-10, lr=5e-4)
    data_train_pathes = params.data_pathes[9:10]
    x_lower_bound = params.x_lower_bound
    kx = params.kx
    ks = params.ks
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)
    
    plt.plot(x1)
    
    from utils import fs_saddle_high_np
    x_init, s_init = x1[0], s1[0]
    x_next, s_next = x_init, s_init
    X_closure = [x_init]
    for dt in dt_x:
        x_next = x_next + fx(1,x_next,s_next)*dt
        s_next = s_next + fs_saddle_high_np(1,x_next,s_next)*dt
        X_closure.append(x_next)
    
    X_closure = np.array(X_closure[:-1])
    
    fig, axes = plt.subplots(1,3, figsize=(20,6))
    axes[0].plot(np.cumsum(dt_x)[:30000], x1[:30000], linewidth=2, label='Monte Carlo data')
    axes[0].plot(np.cumsum(dt_x)[:30000], X_closure[:30000], linewidth=1, c='r', label='Aver. Principle data')
    axes[0].set_xlabel(r'Time t ($\tau_c=1$)')
    axes[0].set_ylabel('Space X')
    axes[0].set_title('Initial')
    axes[0].set_ylim(-0.7,1)
    axes[0].legend()
    
    axes[1].plot(np.cumsum(dt_x)[-10000:-1500], x1[-10000:-1500], linewidth=2)
    axes[1].plot(np.cumsum(dt_x)[-10000:-1500], X_closure[-10000:-1500], linewidth=1, c='r')
    axes[1].yaxis.set_tick_params(labelleft=True)
    axes[1].set_xlabel(r'Time t($\beta J_0 = 0.01, Ext.Pot=5x-1$)')
    axes[1].set_ylim(-0.7,1)
    axes[1].set_title('Remaining...')
    
    axes[2].plot(np.cumsum(dt_x)[-1500:-5], x1[-1500:-5], linewidth=2)
    axes[2].plot(np.cumsum(dt_x)[-1500:-5], X_closure[-1500:-5], linewidth=1, c='r')
    axes[2].yaxis.set_tick_params(labelleft=True)
    axes[2].set_xlabel('Time t')
    axes[2].set_title('Blow-up')
    fig.savefig('./figures/Saddle_ODE1.eps', dpi=300, bbox_inches='tight')

    
    
    fig, axes = plt.subplots(1,2, figsize=(20,5))
    axes[0].plot(np.cumsum(dt_x)[:-1500], x1[:-1500], linewidth=2, label='Monte Carlo data')
    axes[0].plot(np.cumsum(dt_x)[:-1500], X_closure[:-1500], linewidth=1, c='r', label='Aver. Principle data')
    axes[0].set_xlabel(r'Time t ($\tau_c=1$)')
    axes[0].set_ylabel('Space X')
    axes[0].set_title('Initial')
    axes[0].legend()
    
    axes[1].plot(np.cumsum(dt_x)[-1500:-5], x1[-1500:-5], linewidth=2)
    axes[1].plot(np.cumsum(dt_x)[-1500:-5], X_closure[-1500:-5], linewidth=1, c='r')
    axes[1].yaxis.set_tick_params(labelleft=True)
    axes[1].set_xlabel(r'($\beta J_0 = 0.01, Ext.Pot=5x-1$)')
    axes[1].set_title('Blow-up')
    
    fig.savefig('./figures/Saddle_ODE.eps', dpi=300, bbox_inches='tight')
    
    
    
    
    
    
    ###############################
    ###############################
    
    params = parameters(data_type='Saddle_high', n_trainset=20, kx=1, ks=10, x_lower_bound=-1, lr=5e-4)
    data_train_pathes = params.data_pathes[:params.n_trainset]
    x_lower_bound = params.x_lower_bound
    kx = params.kx
    ks = params.ks
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)
    
    
    
    ################################
    ### plot results for model N ###
    ################################
    from model import model_f_saddle
    model_f = model_f_saddle().to(device)
    model_f.load_state_dict(torch.load(params.model_f_path))
    plot_result_N(model_f, x1, s1, diff_x, diff_s)
    
    
        
    ################################
    ### plot results for model H ###
    ################################
    from model import model_std_mean_saddle
    model_H = model_std_mean_saddle().to(device)
    model_H.load_state_dict(torch.load(params.model_std_mean_path))
    
    txs = torch.tensor(np.c_[np.ones_like(s1),x1,s1], dtype=torch.float32).to(device)
    numerical = model_f(txs).cpu().detach().numpy()
    ### s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
    eps = (s2 - (s1 + numerical[:,[-1]]*dt_s))/dt_s**.5
    
    bins_x, bins_s, bins_u = params.bins_x, params.bins_s, params.bins_u
    x_inter, s_inter = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    dist, cond_dist, cond_std, cond_mean, idx_used, unif = \
        get_cond_noise(eps, x1, s1, x_inter, s_inter, shape=[bins_x, bins_s, bins_u], n=100)
        
    x1_re = x1[idx_used,:]
    s1_re = s1[idx_used,:]

    x_idx_ = x1_re[:,[0]]>=x_inter
    x_idx = x_idx_.sum(axis=1)-1

    s_idx_ = s1_re[:,[0]]>=s_inter
    s_idx = s_idx_.sum(axis=1)-1
    
    
    obs_H = np.c_[np.ones_like(s1_re), x1_re, s1_re]
    std_mean = np.c_[cond_std[x_idx, s_idx], cond_mean[x_idx, s_idx]]
    
    idx_ = np.arange(0, obs_H.shape[0], 10)
    obs_H = obs_H[idx_]
    std_mean = std_mean[idx_]
    
    obs_tensor = torch.tensor(obs_H, dtype=torch.float32).to(device)
    targ_out = model_H(obs_tensor).cpu().detach().numpy()
    
    plot_result_H(obs_H, targ_out, std_mean)
    
    
    np.random.seed(85)
    ################################
    ### plot results for model K ###
    ################################
    from model import model_fs_dist_saddle
    ##### compare estimated distribution by NN with empirical distribution
    model = model_fs_dist_saddle().to(device)
    model.load_state_dict(torch.load(params.model_dist_path))
    
    x1_ = x1[idx_used,:]
    s1_ = s1[idx_used,:]
    target_ = dist[idx_used,:]
    from utils import manually_histogram, data_resample_idx
    ### index in each each meshgrid area
    hist2d, hist_ind, _, x_inter, s_inter = manually_histogram(x1_[:,0], s1_[:,0])
    idx_resample = data_resample_idx(hist_ind, num_resample=10)

    x1_resample = x1_[idx_resample]
    s1_resample = s1_[idx_resample]
    # plt.hist2d(x1_resample.flatten(), s1_resample.flatten())
    target_resample = target_[idx_resample]
    
    x1_re = np.repeat(x1_resample, bins_u, axis=0)
    s1_re = np.repeat(s1_resample, bins_u, axis=0)
    unif_re = np.repeat(unif[:,None], x1_resample.shape[0], axis=1).T.reshape(-1,1)

    obs_K = np.c_[x1_re, s1_re, unif_re]
    eps = target_resample.reshape(-1,1)

    obs_tensor = torch.tensor(obs_K, dtype=torch.float32).to(device)
    eps_tensor = torch.tensor(eps, dtype=torch.float32).to(device) 
    
    eps_out = model(obs_tensor).cpu().detach().numpy()
    
    # plot_result_K(obs_K, eps, eps_out)
    plot_result_K_H(obs_K, eps, eps_out, model_H)




    ###################################################################################
    ### compare distribution(train,test,estimated empirical and estimated Gaussian) ###
    ###                     in rare and ordinary case                               ###
    ###################################################################################
    data_test_pathes = params.data_pathes[params.n_trainset:params.n_trainset+80]
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
    
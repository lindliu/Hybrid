#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:33:22 2022

@author: do0236li
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import sys 
sys.path.append("..") 
from torch import nn
from torch import optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy import stats

from utils import get_dataset, get_cond_noise, get_inter_grid
from misc import train_model, parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10)

model_f_path = params.model_f_path
model_std_mean_path = params.model_std_mean_path
model_dist_path = params.model_dist_path
data_pathes = params.data_pathes

data_train_pathes = data_pathes[:params.n_trainset]
data_test_pathes = data_pathes[params.n_trainset:params.n_trainset+1]

bins_x = params.bins_x
bins_s = params.bins_s
bins_u = params.bins_u

x_lower_bound = params.x_lower_bound
kx = params.kx
ks = params.ks

### load model for ODE(x and sigma)
from model import model_f_hopf
model_f = model_f_hopf().to(device)
model_f.load_state_dict(torch.load(model_f_path))

### load model for std and mean
from model import model_std_mean_hopf
model_H = model_std_mean_hopf().to(device)
model_H.load_state_dict(torch.load(model_std_mean_path))

### load model for distribution of noise in sigma
from model import model_fs_dist_hopf
model_K = model_fs_dist_hopf().to(device)
model_K.load_state_dict(torch.load(model_dist_path))


def get_eps(data_pathes, model_f):
    ### load data ###
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)
    
    ### predict numerical derivative
    txs = torch.tensor(np.c_[np.ones_like(s1),x1,s1], dtype=torch.float32).to(device)
    numerical = model_f(txs).cpu().detach().numpy()
    
    ### s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
    eps = (s2 - (s1 + numerical[:,[-1]]*dt_s))/dt_s**.5
    return x1, s1, eps

# x1, s1, eps = get_eps(data_test_pathes, model_f)

def get_eps_out(x1,s1,eps,x_inter_train,s_inter_train,num=1000):
    # x_inter, s_inter = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    target, _, _, _, idx_used, unif = get_cond_noise(eps, x1, s1, x_inter_train, s_inter_train, shape=[bins_x, bins_s, bins_u], n=100)
    
    ##### compare estimated distribution by NN with empirical distribution
    x1_ = x1[:,:]
    s1_ = s1[:,:]
    target_ = target[:,:] # [idx_used,:]
    
    idx = np.random.randint(0,x1_.shape[0],num)
    x1_re = np.repeat(x1_[idx,:], bins_u, axis=0)
    s1_re = np.repeat(s1_[idx,:], bins_u, axis=0)
    unif_re = np.repeat(unif[:,None], idx.shape[0], axis=1).T.reshape(-1,1)
    
    obs = np.c_[x1_re, s1_re, unif_re]
    eps = target_[idx,:].reshape(-1,1)
    
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    eps_tensor = torch.tensor(eps, dtype=torch.float32).to(device) 
    
    eps_out = model_K(obs_tensor).cpu().detach().numpy()
    
    return eps, eps_out, unif, obs_tensor

# eps, eps_out, unif, obs_tensor = get_eps_out(x1,s1,eps,x_inter_train,s_inter_train)
# plt.figure()
# k = 1
# plt.plot(eps_out.flatten()[k*bins_u:(k+1)*bins_u], unif, label='NN estimated distribution')
# plt.plot(eps.flatten()[k*bins_u:(k+1)*bins_u], unif, label='empirical distribution')
# plt.legend()

def get_cons_std(data_train_pathes):
    x1_train, s1_train, eps_train = get_eps(data_train_pathes, model_f)
    x_inter_train, s_inter_train = get_inter_grid(x1_train[:,0], s1_train[:,0], bins_x, bins_s)
    _, _, cond_std, cond_mean, _, _ = get_cond_noise(eps_train, x1_train, s1_train, x_inter_train, s_inter_train, shape=[bins_x, bins_s, bins_u], n=100)
    return cond_std, cond_mean, x_inter_train, s_inter_train

def get_std_mean(cond_std, cond_mean, x_inter, s_inter, obs_tensor, k):
    x_, s_ = obs_tensor[k*bins_u][:-2].cpu().numpy(), obs_tensor[k*bins_u][[-2]].cpu().numpy()
    
    x_idx_ = x_[0]>=x_inter
    x_idx = x_idx_.sum(axis=0)-1
    
    s_idx_ = s_[0]>=s_inter
    s_idx = s_idx_.sum(axis=0)-1
    
    std = cond_std[x_idx,s_idx]
    mean = cond_mean[x_idx,s_idx]
    
    return std, mean

# cond_std, cond_mean, x_inter_train, s_inter_train = get_cons_std(data_train_pathes)
# ### get conditional std of the noise ###
# std, mean = get_std_mean(cond_std, cond_mean, x_inter, s_inter, obs_tensor, k)

# plt.figure()
# plt.plot(eps_out.flatten()[k*bins_u:(k+1)*bins_u], unif, label='NN estimated distribution')
# plt.plot(norm.ppf(unif, loc=mean, scale=std), unif, label="Gaussian distribution")
# plt.plot(eps.flatten()[k*bins_u:(k+1)*bins_u], unif, label='empirical distribution')
# plt.legend()


def get_inverse_cdf(unif, y_pred_, unif_new):
    y_pred = interp1d(unif, y_pred_)

    return y_pred(unif_new)

# k = 1

# ### predict empirical dist
# x1, s1, eps = get_eps(data_test_pathes, model_f)
# eps, eps_out, unif, obs_tensor = get_eps_out(x1,s1,eps, x_inter_train, s_inter_train)
# unif_new = np.linspace(unif.min(), unif.max(), 100)

# y_pred_ = eps_out.flatten()[k*bins_u:(k+1)*bins_u]
# y_pred = get_inverse_cdf(unif, y_pred_, unif_new)
# plt.figure()
# plt.plot(unif_new, y_pred, label='NN estimated distribution')
# plt.legend()


# ### conditional std of the noise ###
# cond_std, cond_mean, x_inter, s_inter = get_cons_std(data_train_pathes)
# std, mean = get_std_mean(cond_std, cond_mean, x_inter, s_inter, obs_tensor, k)
# # input_ = torch.tensor(np.r_[np.array([1]), x1_re[k*50,:], s1_re[k*50,:]], dtype=torch.float32).to(device)
# # out = model_std_mean(input_).cpu().detach().numpy()
# # std, mean = out[0], out[1]
# y_norm_ = norm.ppf(unif, loc=mean, scale=std)
# y_norm = get_inverse_cdf(unif, y_norm_, unif_new)
# plt.plot(unif_new, y_norm, label='normal distribution')
# plt.legend()


# ### empirical dist
# y_emp_ = eps.flatten()[k*bins_u:(k+1)*bins_u]
# y_emp = get_inverse_cdf(unif, y_emp_, unif_new)
# plt.plot(unif_new, y_emp, label='empirical distribution')
# plt.legend()


def Wasserstein_dist(y_pred, y_emp, unif_new, p=1):
    dx = unif_new[1]-unif_new[0]
    dist = (np.sum(np.abs(y_emp-y_pred)**p)*dx)**(1/p)
    return dist

# Wasserstein_dist(y_pred, y_emp, unif_new, p=1)
# Wasserstein_dist(y_norm, y_emp, unif_new, p=1)

def get_cdf(x, y, x_new):
    y_pred = interp1d(x, y)
    return y_pred(x_new)

# cdf_min = max(y_pred.min(), y_norm.min(), y_emp.min())
# cdf_max = min(y_pred.max(), y_norm.max(), y_emp.max())
# x_new = np.linspace(cdf_min, cdf_max, 100)

# y_pred_cdf = get_cdf(y_pred, unif_new, x_new)
# y_norm_cdf = get_cdf(y_norm, unif_new, x_new)
# y_emp_cdf = get_cdf(y_emp, unif_new, x_new)

# plt.plot(x_new, y_pred_cdf)
# plt.plot(x_new, y_norm_cdf)
# plt.plot(x_new, y_emp_cdf)

if __name__=="__main__":
    
    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)
    x_inter_train, s_inter_train = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    x_inter = x_inter_train
    s_inter = s_inter_train
    
    num = 1000 ### to compare distribution of selected num points 
    x1_test, s1_test, eps_test = get_eps(data_test_pathes, model_f)
    x1_mask = np.logical_and(x1_test[:,0]>x_inter.min(), x1_test[:,0]<x_inter.max())
    s1_mask = np.logical_and(s1_test[:,0]>s_inter.min(), s1_test[:,0]<s_inter.max())
    mask = np.logical_and(x1_mask, s1_mask)
    x1_test = x1_test[mask]
    s1_test = s1_test[mask]
    eps_test = eps_test[mask]
    eps, eps_out, unif, obs_tensor = get_eps_out(x1_test,s1_test,eps_test,x_inter,s_inter,num)
    unif_new = np.linspace(unif.min(), unif.max(), 100)
    
    W1_dist_pred, W1_dist_norm = [], []
    W2_dist_pred, W2_dist_norm = [], []
    KL_dist_pred, KL_dist_norm = [], []
    TA_dist_pred, TA_dist_norm = [], []
    for k in range(num):
        y_pred_ = eps_out.flatten()[k*bins_u:(k+1)*bins_u]
        y_pred = get_inverse_cdf(unif, y_pred_, unif_new)
        
        point = torch.cat([torch.tensor([1]).to(device), obs_tensor[k*bins_u, [0,1,2]]])
        std, mean = model_H(point).cpu().detach().numpy().flatten()
        std = np.abs(std)
        y_norm_ = norm.ppf(unif, loc=mean, scale=std)
        # y_norm_ = norm.ppf(unif, loc=0, scale=std)
        y_norm = get_inverse_cdf(unif, y_norm_, unif_new)
        
        y_emp_ = eps.flatten()[k*bins_u:(k+1)*bins_u]
        y_emp = get_inverse_cdf(unif, y_emp_, unif_new)
        
        ### get Wasserstein distance
        p = 1
        W1_dist_pred.append(Wasserstein_dist(y_pred, y_emp, unif_new, p))
        W1_dist_norm.append(Wasserstein_dist(y_norm, y_emp, unif_new, p))
        ### get Wasserstein distance
        p = 2
        W2_dist_pred.append(Wasserstein_dist(y_pred, y_emp, unif_new, p))
        W2_dist_norm.append(Wasserstein_dist(y_norm, y_emp, unif_new, p))
        
        
    print('W1 estimated normal distance: {}'.format(np.mean(W1_dist_norm)))
    print('W1 estimated empirical distance: {}'.format(np.mean(W1_dist_pred)))

    print('W2 estimated normal distance: {}'.format(np.mean(W2_dist_norm)))
    print('W2 estimated empirical distance: {}'.format(np.mean(W2_dist_pred)))

    
    plt.figure()
    plt.scatter(np.arange(num), W1_dist_pred, s=1, label='W Learned dist')
    plt.scatter(np.arange(num), W1_dist_norm, s=1, label='W Normal dist')
    plt.legend()
    
    
    plt.figure(figsize=[10,6])
    plt.plot(y_pred, unif_new, '--', label='estimated empirical dist')
    plt.plot(y_norm, unif_new, '--', label='estimated normal dist')
    plt.plot(y_emp, unif_new, label='testset empirical dist')
    plt.legend()
    # plt.savefig('./figures/Hopf_dist_slides.png',bbox_inches='tight')
    
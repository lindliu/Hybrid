#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:07:22 2022

Compare E-MPL with Gaussian noise and E-MPL with empirical noise

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append("..") 
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

from misc import parameters
from utils import get_dataset, get_cond_noise, get_inter_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='Saddle_high', n_trainset=20, kx=1, ks=10, x_lower_bound=-1, lr=5e-4)

data_train_pathes = params.data_pathes[:params.n_trainset]
data_test_pathes = params.data_pathes[params.n_trainset:params.n_trainset+80]

bins_x = params.bins_x
bins_s = params.bins_s
bins_u = params.bins_u

x_lower_bound = params.x_lower_bound
kx = params.kx
ks = params.ks

### load model for ODE(x and sigma)
from model import model_f_saddle
model_f = model_f_saddle().to(device)
model_f.load_state_dict(torch.load(params.model_f_path))

### load model for std and mean
from model import model_std_mean_saddle
model_H = model_std_mean_saddle().to(device)
model_H.load_state_dict(torch.load(params.model_std_mean_path))

### load model for distribution of noise in sigma
from model import model_fs_dist_saddle
model_K = model_fs_dist_saddle().to(device)
model_K.load_state_dict(torch.load(params.model_dist_path))


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


def get_distribution(x,s,unif):
    x_obs = np.repeat(x, bins_u, axis=0)
    s_obs = np.repeat(s, bins_u, axis=0)
    unif_obs = np.repeat(unif[:,None], s.shape[0], axis=1).T.reshape(-1,1)
    
    obs = np.c_[x_obs, s_obs, unif_obs]    
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    
    emp_estimated = model_K(obs_tensor).cpu().detach().numpy()
    
    return obs_tensor.cpu().numpy(), emp_estimated

def get_cons_std(data_train_pathes):
    x1_train, s1_train, eps_train = get_eps(data_train_pathes, model_f)
    x_inter_train, s_inter_train = get_inter_grid(x1_train[:,0], s1_train[:,0], bins_x, bins_s)
    _, _, cond_std, cond_mean, _, unif = get_cond_noise(eps_train, x1_train, s1_train, x_inter_train, s_inter_train, shape=[bins_x, bins_s, bins_u])
    return cond_std, cond_mean, x_inter_train, s_inter_train, unif


def get_samples_grid(eps, x1, s1, x_inter, s_inter, shape, n=30):
    bins_x, bins_s, bins_u = shape

    samp = []
    for i in range(bins_x):
        samp.append([])
        for j in range(bins_s):
            x_idx_ = x1[:,[0]]>=x_inter
            x_idx = x_idx_.sum(axis=1)-1
    
            s_idx_ = s1[:,[0]]>=s_inter
            s_idx = s_idx_.sum(axis=1)-1
            
            both_idx = np.logical_and(x_idx==i, s_idx==j)
            
            if both_idx.sum()>n: ## less than 100 data has no statistical meaning
                eps_sample = eps.flatten()[both_idx]
                samp[-1].append(eps_sample)
            else:
                samp[-1].append([])
        print('{}/{} has done!'.format(i,bins_x))
        
    return samp

from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
###https://stackoverflow.com/questions/44132543/python-inverse-empirical-cumulative-distribution-function-ecdf
def inverted_edf(unif, eps_sample):
    sample_edf = edf.ECDF(eps_sample)
    
    slope_changes = sorted(set(eps_sample))
    
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    
    slope_changes.insert(0,2*slope_changes[0]-slope_changes[1])
    sample_edf_values_at_slope_changes.insert(0,0)
    
    return sample_edf_values_at_slope_changes, slope_changes

def get_inverted_cdf(x, y, unif):
    y_pred = interp1d(x, y)
    return y_pred(unif)

def Wasserstein_dist(f1, f2, unif, p=1):
    dx = unif[1]-unif[0]
    dist = ( np.sum(np.abs(f1-f2)**p, axis=1)*dx )**(1/p)
    return dist


if __name__=="__main__":

    x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
        get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)
    x_inter_train, s_inter_train = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    x_inter = x_inter_train
    s_inter = s_inter_train
    
    middle_x = x_inter[:-1]+(x_inter[1]-x_inter[0])/2
    middle_s = s_inter[:-1]+(s_inter[1]-s_inter[0])/2
    
    x_, s_ = np.meshgrid(middle_x,middle_s,indexing='ij')
    unif = np.linspace(0,1,bins_u+2)[1:-1]
    obs, emp_estimated = get_distribution(x_.flatten(), s_.flatten(), unif)
    emp_train = emp_estimated.reshape([bins_x,bins_s,bins_u])
    
    x1_test, s1_test, eps_test = get_eps(data_test_pathes, model_f)
    samp = get_samples_grid(eps_test, x1_test, s1_test, x_inter, s_inter, shape=[bins_x, bins_s, bins_u], n=200)
    
    norm_train = np.zeros([bins_x, bins_s, bins_u])
    emp_test = np.zeros([bins_x, bins_s, bins_u])
    compare = []
    for i in range(len(samp)):
        for j in range(len(samp[1])):
            samples = samp[i][j]
            if len(samples)>0:
                compare.append([])
                
                point = torch.tensor(np.c_[1, middle_x[i], middle_s[j]],dtype=torch.float32).to(device)
                std, mean = model_H(point).cpu().detach().numpy().flatten()
                norm_train[i,j,:] = norm.ppf(unif, loc=mean, scale=std)
                
                unif_, samples_ = inverted_edf(unif, samples)
                emp_test[i,j,:] = get_inverted_cdf(unif_, samples_, unif)
            
                compare[-1].append(emp_train[i,j,:])
                compare[-1].append(norm_train[i,j,:])
                compare[-1].append(emp_test[i,j,:])
                
    compare = np.array(compare)
    
    idx = 5
    plt.figure(figsize=[10,8])
    fig = plt.plot(unif, compare[idx,:,:].T)
    plt.legend(fig, ['empirical i-cdf train','normal i-cdf train','empirical i-cdf test'])
    plt.title('inverse cdf comparison')

    p = 1
    W_dist_norm = Wasserstein_dist(compare[:,1,:], compare[:,2,:], unif, p)
    W_dist_pred = Wasserstein_dist(compare[:,0,:], compare[:,2,:], unif, p)
    print('W{} normal distance: {}'.format(p, np.mean(W_dist_norm)))
    print('W{} empirical distance: {}'.format(p, np.mean(W_dist_pred)))

    p = 2
    W_dist_norm = Wasserstein_dist(compare[:,1,:], compare[:,2,:], unif, p)
    W_dist_pred = Wasserstein_dist(compare[:,0,:], compare[:,2,:], unif, p)
    print('W{} normal distance: {}'.format(p, np.mean(W_dist_norm)))
    print('W{} empirical distance: {}'.format(p, np.mean(W_dist_pred)))
    
    # plt.figure()
    # plt.scatter(np.arange(W_dist_pred.shape[0]), W_dist_pred, s=1, label='W predict dist')
    # plt.scatter(np.arange(W_dist_norm.shape[0]), W_dist_norm, s=1, label='W normal dist')
    # plt.legend()
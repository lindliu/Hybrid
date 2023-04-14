#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:52:04 2022

@author: dliu
"""

import pandas as pd
import numpy as np
import torch


def get_idx(x, sigma, x_lower_bound=-1, s_lower_bound=-10):
    """
    Input: 
        x: 1d data in [-inf, 1]
        sigma: 1d data in [0,1]
        
    Output: 
        the minimum index that all x>x_lower_bound and sigma>s_lower_bound
    """
    x_idx = []
    for i in range(x.shape[1]):
        if False not in list(x[:,i]>x_lower_bound):
            x_idx_ = x.shape[0]
        else:
            x_idx_ = list(x[:,i]>x_lower_bound).index(False)
        x_idx.append(x_idx_)
    x_idx = min(x_idx)
    
    sigma = sigma.flatten()
    if False not in list(sigma>s_lower_bound):
        s_idx = x.shape[0]
    else:
        s_idx = list(sigma>s_lower_bound).index(False)
        
    idx = min(x_idx, s_idx)
    return idx

def collect_datasets(data_pathes, x_lower_bound=-1):
    
    t_several, x_several, sigma_several = [], [], []
    for i in range(len(data_pathes)):
        data_var = pd.read_csv(data_pathes[i], delimiter=' ',header=None).to_numpy()
    
        t_ = data_var[:,[0]]
        x_ = data_var[:,1:-1]
        sigma_ = data_var[:,[-1]]
    
        idx = get_idx(x_,sigma_,x_lower_bound)
        print("Take first {} data points from {}".format(idx, x_.shape))
        t_ = t_[:idx]
        x_ = x_[:idx]
        sigma_ = sigma_[:idx]
    
        t_several.append(t_)
        x_several.append(x_)
        sigma_several.append(sigma_)

    return t_several, x_several, sigma_several

def collect_datasets_diff(t_several, x_several, sigma_several, kx=1, ks=1):
    """
    x1,x2 and s1,s2 is for calculating numerical derivative
    diff_x = (x2-x1)/dt
    diff_s = (s2-s1)/dt
    
    if we are trying to calculate diff_x, then kx=1, since x data has no noise.
    """
    assert kx<=ks, "kx should no lager than ks, since x has no noise"
    
    x1, s1 = [], []
    x2, s2 = [], []
    dt_x, dt_s = [], []
    for ttt, xxx, sss in zip(t_several, x_several, sigma_several):
        dt_s.extend(list(ttt[ks:,:] - ttt[:-ks,:]))
        s1.extend(list(sss[:-ks,:]))
        s2.extend(list(sss[ks:,:]))
        
        if -ks+kx != 0:
            dt_x.extend(list((ttt[kx:,:] - ttt[:-kx,:])[:-ks+kx]))
            x1.extend(list((xxx[:-kx,:])[:-ks+kx,:]))
            x2.extend(list((xxx[kx:,:])[:-ks+kx,:]))
        else:
            dt_x.extend(list((ttt[kx:,:] - ttt[:-kx,:])))
            x1.extend(list((xxx[:-kx,:])))
            x2.extend(list((xxx[kx:,:])))
        
    x1, s1 = np.array(x1), np.array(s1)
    x2, s2 = np.array(x2), np.array(s2)
    dt_x = np.array(dt_x)
    dt_s = np.array(dt_s)
    
    return x1, x2, s1, s2, dt_x, dt_s

def get_inter_grid(x, s, bins_x, bins_s):
    x_inter = np.linspace(x.min(), x.max()+10e-5,bins_x+1) #add 10e-5 to make sure end point will be included
    s_inter = np.linspace(s.min(), s.max()+10e-5,bins_s+1)
    return x_inter, s_inter

def manually_histogram(x, sigma, bins_x=10, bins_s=10):
    """
    x and sigma are 1d dataset
    """
    #find data index in each area
    ind = np.arange(x.shape[0])
    
    x_inter, s_inter = get_inter_grid(x, sigma, bins_x, bins_s)
    
    hist2d = np.zeros([bins_s,bins_x]) ##number of data in each sub area
    coor_in_hist = np.zeros([x.shape[0], 2], dtype=np.int)
    hist_ind = [] ##index in each sub area
    for i in range(bins_s):
        hist_ind.append([])
        # hist_ind_bool.append([])
        for j in range(bins_x):
            x_in = np.logical_and(x_inter[j]<=x, x<x_inter[j+1])
            s_in = np.logical_and(s_inter[i]<=sigma, sigma<s_inter[i+1])
    
            index_ = np.logical_and(x_in,s_in)
            hist2d[i,j] = np.sum(index_)
            
            coor_in_hist[index_, 0] = j
            coor_in_hist[index_, 1] = i
            
            hist_ind[-1].append(ind[index_])
    
    return hist2d, hist_ind, coor_in_hist, x_inter, s_inter
    

def manually_histogram_(x, sigma, bins_x=10, bins_s=10):
    """
    x and sigma are 1d dataset
    """
    #find data index in each area
    ind = np.arange(x.shape[0])
    
    x_inter, s_inter = get_inter_grid(x, sigma, bins_x, bins_s)
    
    hist2d = np.zeros([bins_x,bins_s]) ##number of data in each sub area
    coor_in_hist = np.zeros([x.shape[0], 2], dtype=np.int32)
    hist_ind = [] ##index in each sub area
    for i in range(bins_x):
        hist_ind.append([])
        # hist_ind_bool.append([])
        for j in range(bins_s):
            x_in = np.logical_and(x_inter[i]<=x, x<x_inter[i+1])
            s_in = np.logical_and(s_inter[j]<=sigma, sigma<s_inter[j+1])
    
            index_ = np.logical_and(x_in,s_in)
            hist2d[i,j] = np.sum(index_)
            
            coor_in_hist[index_, 0] = i
            coor_in_hist[index_, 1] = j
            
            hist_ind[-1].append(ind[index_])
    
    return hist2d, hist_ind, coor_in_hist, x_inter, s_inter


def data_resample_idx(hist_ind, num_resample=10000):
    """
    get resample index
    num_resample is resample number in each histogram area
    """
    bins_s = len(hist_ind)
    bins_x = len(hist_ind[0])
    
    ###resample in each area
    idx_resample = []
    for i in range(bins_s):
        for j in range(bins_x):
            if len(hist_ind[i][j]) != 0:
                idx_resample_ = np.random.choice(hist_ind[i][j], num_resample)
                idx_resample.append(idx_resample_)
    
    return np.array(idx_resample).flatten()
    
    
    

def get_dataset(data_pathes, x_lower_bound=-1, kx=1, ks=1, resample=False):
    assert ks>=kx, "ks should lager than kx"
    
    datasets_all = collect_datasets(data_pathes, x_lower_bound=x_lower_bound)
    t_several, x_several, sigma_several = datasets_all

    x1, x2, s1, s2, dt_x, dt_s = collect_datasets_diff(t_several, x_several, sigma_several, kx=kx, ks=ks)
    diff_x = (x2-x1)/dt_x
    diff_s = (s2-s1)/dt_s
    
    if not resample:
        return x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s
    
    else:
        ### index in each each meshgrid area
        hist2d, hist_ind, _, x_inter, s_inter = manually_histogram(x1[:,0], s1[:,0])
        
        ### resample in each area
        idx_resample = data_resample_idx(hist_ind, num_resample=1000)
        
        x1_resample = x1[idx_resample]
        x2_resample = x2[idx_resample]
        s1_resample = s1[idx_resample]
        s2_resample = s2[idx_resample]
        diff_x_resample = diff_x[idx_resample]
        diff_s_resample = diff_s[idx_resample]
        
        dt_x_resample = dt_x[idx_resample]
        dt_s_resample = dt_s[idx_resample]
        return x1_resample, x2_resample, s1_resample, s2_resample, diff_x_resample, diff_s_resample, dt_x_resample, dt_s_resample
        

from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
###https://stackoverflow.com/questions/44132543/python-inverse-empirical-cumulative-distribution-function-ecdf
def inverted_edf(unif, eps_sample):
    sample_edf = edf.ECDF(eps_sample)
    
    slope_changes = sorted(set(eps_sample))
    
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    
    slope_changes.insert(0,2*slope_changes[0]-slope_changes[1])
    sample_edf_values_at_slope_changes.insert(0,0)
    
    inverted = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    
    y = inverted(unif)
    return y

def get_cond_noise(eps, x1, s1, x_inter, s_inter, shape, n=30):
    """
    Parameters
    ----------
    eps : TYPE
        DESCRIPTION.
    x1 : TYPE
        DESCRIPTION.
    s1 : TYPE
        DESCRIPTION.
    x_inter : TYPE
        DESCRIPTION.
    s_inter : TYPE
        DESCRIPTION.
    shape : TYPE
        DESCRIPTION.
    n : int
        The minimum number of data in each grid, if it's less than n, 
        we won't keep the empirical distribution in that grid.
        Thus, n matters for rare event.
        The default is 30.

    Returns
    -------
    target : TYPE
        DESCRIPTION.
    cond_dist : TYPE
        DESCRIPTION.
    cond_std : TYPE
        DESCRIPTION.
    cond_mean : TYPE
        DESCRIPTION.
    idx_used : TYPE
        DESCRIPTION.
    unif : TYPE
        DESCRIPTION.

    """
    bins_x, bins_s, bins_u = shape
    # x_inter, s_inter = get_inter_grid(x1[:,0], s1[:,0], bins_x, bins_s)
    
    unif = np.linspace(0,1,bins_u+2)[1:-1]
    
    ### calculate varivance of numerical fxs ###
    cond_dist = np.zeros([bins_x,bins_s,bins_u])
    cond_std = np.zeros([bins_x,bins_s])
    cond_mean = np.zeros([bins_x,bins_s])
    target = np.zeros([x1.shape[0], bins_u])
    idx_used = np.zeros(x1.shape[0])
    
    ij = np.zeros([bins_x*bins_s, 2], dtype=np.int16)
    k = 0
    for i in range(bins_x):
        for j in range(bins_s):
            x_idx_ = x1[:,[0]]>=x_inter
            x_idx = x_idx_.sum(axis=1)-1
    
            s_idx_ = s1[:,[0]]>=s_inter
            s_idx = s_idx_.sum(axis=1)-1
            
            both_idx = np.logical_and(x_idx==i, s_idx==j)
            
            if both_idx.sum()>n: ## less than 100 data has no statistical meaning
                eps_sample = eps.flatten()[both_idx]
                cond_std[i,j] = np.std(eps_sample)
                cond_mean[i,j] = np.mean(eps_sample)
                
                distribution = inverted_edf(unif, eps_sample)
                cond_dist[i,j,:] = distribution
                
                target[both_idx,:] = np.repeat(distribution[None,:],both_idx.sum(),axis=0)
                idx_used = np.logical_or(idx_used, both_idx)
            else:
                ij[k,:] = i,j
                k += 1
                # continue
        print('{}/{} has done!'.format(i,bins_x))
        
    ### fill in empty place with average of std and mean
    mask = np.ones_like(cond_std, dtype=bool)
    mask[ij[:,0], ij[:,1]] = False
    cond_std[~mask] = np.mean(cond_std[mask])
    cond_mean[~mask] = np.mean(cond_mean[mask])
        
    return target, cond_dist, cond_std, cond_mean, idx_used, unif


def fx(t,x,sigma):
    """
    Real ODE of x for Saddle with high noise case 

    Input: t,x,sigma
        
    tauc=.1;  gamma=-0.05; z=0.5; b=1; c=5; h0=-1;
    x' = b/tauc * (z-sigma) + gamma/tauc * x^2
    """
    tauc=.1;  gamma=-0.05; z=0.5; b=1;
    
    f = b/tauc * (z-sigma) + gamma/tauc * x**2
    return f

def fs_saddle_high_np(t,x,sigma):
    """
    Mean field ODE of sigma for Saddle with high noise case 
    
    Input: t,x,sigma
    
    tauI=1; beta=.01; J=1; c=5; h0=-1
    sigma' = b/tauI * (1-sigma-sigma*exp(-beta*J*sigma)*exp(-beta*(c*x+h0)))
    """
    tauI=1; beta=.01; J=1; c=5; h0=-1
    f = 1/tauI * (1-sigma-sigma*np.exp(-beta*J*sigma)*np.exp(-beta*(c*x+h0)))
    return f

def fs_saddle_high(t,x,sigma):
    """
    Mean field ODE of sigma for Saddle with high noise case 

    Input: t,x,sigma
    
    tauI=1; beta=.01; J=1; c=5; h0=-1
    sigma' = b/tauI * (1-sigma-sigma*exp(-beta*J*sigma)*exp(-beta*(c*x+h0)))
    """
    tauI=1; beta=.01; J=1; c=5; h0=-1
    f = 1/tauI * (1-sigma-sigma*torch.exp(-beta*J*sigma)*torch.exp(-beta*(c*x+h0)))
    return f

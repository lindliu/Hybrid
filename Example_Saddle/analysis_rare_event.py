#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:54:16 2022

This code has two parts:
    Part I: Simulate 100 times by two E-MPL models.
    Part II: Compare the distribution of rare event time.
@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append("..") 
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import collect_datasets
from misc import parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='Saddle_high', n_trainset=20, kx=1, ks=10, x_lower_bound=-1, lr=5e-4)


########################## Part I #############################

###############################################################
##### analysis the distribution of the time of rare event #####
###############################################################
x_lower_bound = params.x_lower_bound
kx = params.kx
ks = params.ks

datasets_all = collect_datasets(params.data_pathes, x_lower_bound=x_lower_bound)
t_several, x_several, sigma_several = datasets_all

t_end_ = np.array([i[-1] for i in t_several])
x_end_ = np.array([i[-1] for i in x_several])
np.savetxt('./figures/rare_event_data.txt', t_end_)
plt.hist(t_end_)
plt.title('histogram of rare event time point by 100 datasets')
# plt.savefig('./figures/dist_by_data.png')




### load model for ODE(x and sigma)
from model import model_f_saddle
model_f = model_f_saddle().to(device)
model_f.load_state_dict(torch.load(params.model_f_path))

########################################################
###### simulate both x and s with Gaussian noise #######
########################################################
from simulate_tools import simulate_xs_with_GNN
from model import model_std_mean_saddle
model_H = model_std_mean_saddle().to(device)
model_H.load_state_dict(torch.load(params.model_std_mean_path))

length = 10000
idx_init = 0
dt = 0.001*ks
x_n_all, s_n_all, t_n_all = [], [], []
for i in range(100):
    x_init, s_init = [[0.9994]], [[0.503]]
    x_series, s_series, t_series = \
        simulate_xs_with_GNN(model_f, model_H, x_init, s_init, dt, length, stop_threshold=x_lower_bound)
    x_n_all.append(x_series)
    s_n_all.append(s_series)
    t_n_all.append(t_series)
    print(i)
# l = 1
# plt.plot(t_n_all[l], x_n_all[l])
t_end_n = np.array([i[-1] for i in t_n_all])
np.savetxt('./figures/rare_event_normal.txt', t_end_n)
plt.figure()
plt.hist(t_end_n)
plt.title('histogram by Gaussian noise')
# plt.savefig('./figures/dist_by_GNN.png')



#########################################################
###### simulate both x and s with empirical noise #######
#########################################################
from simulate_tools import simulate_xs_with_EN
from model import model_fs_dist_saddle
model_K = model_fs_dist_saddle().to(device)
model_K.load_state_dict(torch.load(params.model_dist_path))

length = 10000
idx_init = 0
dt = 0.001*ks
x_e_all, s_e_all, t_e_all = [], [], []
for i in range(100):
    x_init, s_init = [[0.9994]], [[0.503]]
    x_series, s_series, t_series = \
        simulate_xs_with_EN(model_f, model_K, x_init, s_init, dt, length=length, stop_threshold=x_lower_bound, lamb=0.493) ## lamb=0.5
    x_e_all.append(x_series)
    s_e_all.append(s_series)
    t_e_all.append(t_series)
    print(i)
# l = 1
# plt.plot(t_e_all[l], x_e_all[l])
t_end_e = np.array([i[-1] for i in t_e_all])
np.savetxt('./figures/rare_event_empirical.txt', t_end_e)
plt.figure()
plt.hist(t_end_e)
plt.title('histogram by Empirical noise')
# plt.savefig('./figures/dist_by_EN.png')
    






####################################################
###################### Part II #####################
##### compare the time that rare event happens #####
####################################################

from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
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
    dist = ( np.sum(np.abs(f1-f2)**p)*dx )**(1/p)
    return dist


t_end_ = np.loadtxt('./figures/rare_event_data.txt')
t_end_n = np.loadtxt('./figures/rare_event_normal.txt')
t_end_e = np.loadtxt('./figures/rare_event_empirical.txt')
# time_rare_event = np.sort(np.c_[t_end_, t_end_n, t_end_e], axis=0)
# plt.plot(time_rare_event/100)

unif = np.linspace(0,1,100)
unif_, t_end_ = inverted_edf(unif, t_end_)
t_end_ = get_inverted_cdf(unif_, t_end_, unif)

unif_, t_end_n = inverted_edf(unif, t_end_n)
t_end_n = get_inverted_cdf(unif_, t_end_n, unif)

unif_, t_end_e = inverted_edf(unif, t_end_e)
t_end_e = get_inverted_cdf(unif_, t_end_e, unif)

p = 1
W_dist_norm = Wasserstein_dist(t_end_n, t_end_, unif, p)
W_dist_pred = Wasserstein_dist(t_end_e, t_end_, unif, p)

print('distance from E-MLP with Gaussian noise to test is: {}'.format(W_dist_norm))
print('distance from E-MLP with empirical noise to test is: {}'.format(W_dist_pred))


fig = plt.figure(figsize=[10,5], constrained_layout=True)
plt.plot(t_end_, unif, linewidth=2)
plt.plot(t_end_n, unif, '--', linewidth=2)
plt.plot(t_end_e, unif, '--', linewidth=2)
plt.legend(['By Saddle(high noise) data','By simulations of E-MLP with Gaussian noise','By simulations of E-MLP with empirical noise'])
plt.title('The ECDF of the time that rare event happens')
plt.xlabel('t')
# plt.savefig('./figures/distribution_rare_event.png', bbox_inches='tight')


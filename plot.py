#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:36:33 2022

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib
font = {'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


Hopf_data = np.load('./Example_Hopf/figures/Hopf_part.npy')
Hopf_simulation = np.load('./Example_Hopf/figures/Hopf_simulation.npy')

Saddle_high_data = np.load('./Example_Saddle/figures/Saddle_high_part.npy')
Saddle_high_simulation_ = np.load('./Example_Saddle/figures/Saddle_high_simulations.npz')
Saddle_high_s = [Saddle_high_simulation_[key] for key in Saddle_high_simulation_]

Saddle_low_data = np.load('./Example_Saddle/figures/Saddle_low_part.npy')
Saddle_low_simulation_ = np.load('./Example_Saddle/figures/Saddle_low_simulations.npz')
Saddle_low_s = [Saddle_low_simulation_[key] for key in Saddle_low_simulation_]


fig = plt.figure(figsize=[15,10], constrained_layout=True)
fig.suptitle('CGL')
subfigs = fig.subfigures(nrows=3, ncols=1)

row1 = subfigs[0]
# row1.suptitle('CGL')
axs = row1.subplots(nrows=1, ncols=3)
ax1,ax2,ax3 = axs

ax1.plot(Hopf_data[:,0], Hopf_data[:,1], linewidth=2, label='$X_1$')
ax1.plot(Hopf_simulation[0][:,0], Hopf_simulation[0][:,1], '--', linewidth=1)
ax1.plot(Hopf_simulation[1][:,0], Hopf_simulation[1][:,1], '--', linewidth=1)
ax1.plot(Hopf_simulation[2][:,0], Hopf_simulation[2][:,1], '--', linewidth=1)
# ax1.legend(loc=3)
ax1.set_xlabel('t')
ax1.set_ylabel('$X_1$')
# ax1.set_xticks([])
ax1.set_yticks([])

ax2.plot(Hopf_data[:,0], Hopf_data[:,2], linewidth=2, label='$X_2$')
ax2.plot(Hopf_simulation[0][:,0], Hopf_simulation[0][:,2], '--', linewidth=1)
ax2.plot(Hopf_simulation[1][:,0], Hopf_simulation[1][:,2], '--', linewidth=1)
ax2.plot(Hopf_simulation[2][:,0], Hopf_simulation[2][:,2], '--', linewidth=1)
# ax2.legend(loc=3)
ax2.set_xlabel('t')
ax2.set_ylabel('$X_2$')
# ax2.set_xticks([])
ax2.set_yticks([])

ax3.plot(Hopf_data[:,0], Hopf_data[:,3], linewidth=2, label=r'$\bar{\sigma}$', c='tab:blue')
ax3.plot(Hopf_simulation[0][:,0], Hopf_simulation[0][:,3], '--', linewidth=.5)
ax3.plot(Hopf_simulation[1][:,0], Hopf_simulation[1][:,3], '--', linewidth=.5)
ax3.plot(Hopf_simulation[2][:,0], Hopf_simulation[2][:,3], '--', linewidth=.5)
# ax3.legend(loc=3)
ax3.set_xlabel('t')
ax3.set_ylabel(r'$\bar{\sigma}$')
# ax3.set_xticks([])
ax3.set_yticks([])


row2 = subfigs[1]
row2.suptitle('Saddle(low noise)')
axs = row2.subplots(nrows=1, ncols=2)
ax4,ax5 = axs

ax4.plot(Saddle_low_data[:,0], Saddle_low_data[:,1], linewidth=2, label='$X$')
ax4.plot(Saddle_low_s[0][:,0], Saddle_low_s[0][:,1], '--', linewidth=1)
# ax4.plot(Saddle_low_s[1][:,0], Saddle_low_s[1][:,1], '--', linewidth=1)
ax4.plot(Saddle_low_s[2][:,0], Saddle_low_s[2][:,1], '--', linewidth=1)
# ax4.plot(Saddle_low_s[3][:,0], Saddle_low_s[3][:,1], '--', linewidth=1)
ax4.plot(Saddle_low_s[4][:,0], Saddle_low_s[4][:,1], '--', linewidth=1)
ax4.plot(Saddle_low_s[5][:,0], Saddle_low_s[5][:,1], '--', linewidth=1)
# ax4.legend(loc=3)
# ax4.axis('off')
ax4.set_xlabel('t')
ax4.set_ylabel('$X$')
ax4.set_yticks([0.0,0.5,1.0])
ax4.set_yticklabels([0.0,0.5,1.0])
# ax4.set_xticks([])
# ax4.set_yticks([])

ax5.plot(Saddle_low_data[:,0], Saddle_low_data[:,2], linewidth=2, label=r'$\bar{\sigma}$')
ax5.plot(Saddle_low_s[0][:,0], Saddle_low_s[0][:,2], '--', linewidth=.5)
ax5.plot(Saddle_low_s[1][:,0], Saddle_low_s[1][:,2], '--', linewidth=.5)
ax5.plot(Saddle_low_s[2][:,0], Saddle_low_s[2][:,2], '--', linewidth=.5)
ax5.plot(Saddle_low_s[3][:,0], Saddle_low_s[3][:,2], '--', linewidth=.5)
ax5.plot(Saddle_low_s[4][:,0], Saddle_low_s[4][:,2], '--', linewidth=.5)
# ax5.plot(Saddle_low_s[5][:,0], Saddle_low_s[5][:,2], '--', linewidth=.5)
# ax5.legend(loc=3)
ax5.set_xlabel('t')
ax5.set_ylabel(r'$\bar{\sigma}$')
ax5.set_yticks([0.45, .5, .55])
ax5.set_yticklabels([0.45, .5, .55])
# ax5.set_xticks([])
# ax5.set_yticks([])


row3 = subfigs[2]
row3.suptitle('Saddle(high noise)')
axs = row3.subplots(nrows=1, ncols=2)
ax6,ax7 = axs

ax6.plot(Saddle_high_data[:,0], Saddle_high_data[:,1], linewidth=2, label='$X$')
ax6.plot(Saddle_high_s[0][:,0], Saddle_high_s[0][:,1], '--', linewidth=1)
ax6.plot(Saddle_high_s[1][:,0], Saddle_high_s[1][:,1], '--', linewidth=1)
ax6.plot(Saddle_high_s[2][:,0], Saddle_high_s[2][:,1], '--', linewidth=1)
ax6.plot(Saddle_high_s[3][:,0], Saddle_high_s[3][:,1], '--', linewidth=1)
ax6.plot(Saddle_high_s[4][:,0], Saddle_high_s[4][:,1], '--', linewidth=1)
# ax6.plot(Saddle_high_s[5][:,0], Saddle_high_s[5][:,1], '--', linewidth=1)
# ax6.legend(loc=3)
# ax6.axis('off')
ax6.set_xlabel('t')
ax6.set_ylabel('$X$')
ax6.set_yticks([-1.0,0,1.0])
ax6.set_yticklabels([-1.0,0,1.0])
# ax6.set_xticks([])
# ax6.set_yticks([])

ax7.plot(Saddle_high_data[:,0], Saddle_high_data[:,2], linewidth=2, label=r'$\bar{\sigma}$')
ax7.plot(Saddle_high_s[0][:,0], Saddle_high_s[0][:,2], '--', linewidth=.5)
ax7.plot(Saddle_high_s[1][:,0], Saddle_high_s[1][:,2], '--', linewidth=.5)
ax7.plot(Saddle_high_s[2][:,0], Saddle_high_s[2][:,2], '--', linewidth=.5)
ax7.plot(Saddle_high_s[3][:,0], Saddle_high_s[3][:,2], '--', linewidth=.5)
ax7.plot(Saddle_high_s[4][:,0], Saddle_high_s[4][:,2], '--', linewidth=.5)
# ax7.plot(Saddle_high_s[5][:,0], Saddle_high_s[5][:,2], '--', linewidth=.5)
# ax7.legend(loc=3)
ax7.set_xlabel('t')
ax7.set_ylabel(r'$\bar{\sigma}$')
ax7.set_yticks([0.45, .5, .55])
ax7.set_yticklabels([0.45, .50, .55])
# ax7.set_xticks([])
# ax7.set_yticks([])

# fig.savefig('./figures/data_and_simulations.eps', dpi=300, bbox_inches='tight')









import os
import glob
from utils import collect_datasets

data_pathes = glob.glob('././datasets/Saddle_datasets_high_noise/*data')
data_pathes = sorted(data_pathes, key=lambda x: int(os.path.split(x)[1].split('_')[0]))

datasets_all = collect_datasets(data_pathes, x_lower_bound=-1)
t_several, x_several, sigma_several = datasets_all

plt.figure()
for i in range(100):
    plt.plot(t_several[i],x_several[i])
    




##### compare the time that rare event happens #####

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


t_end_ = np.loadtxt('./Example_Saddle/figures/rare_event_data.txt')
t_end_n = np.loadtxt('./Example_Saddle/figures/rare_event_normal.txt')
t_end_e = np.loadtxt('./Example_Saddle/figures/rare_event_empirical.txt')
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
    
    
fig = plt.figure(figsize=[10,5], constrained_layout=True)
plt.plot(t_end_, unif, linewidth=2)
plt.plot(t_end_n, unif, '--', linewidth=2)
plt.plot(t_end_e, unif, '--', linewidth=2)
plt.legend(['By Saddle(high noise) data','By simulations of E-MLP with Gaussian noise','By simulations of E-MLP with empirical noise'])
plt.title('The ECDF of the time that rare event happens')
plt.xlabel('t')
# plt.savefig('./figures/distribution_rare_event.eps', dpi=300, bbox_inches='tight')
           




fig, axes = plt.subplots(1,2, figsize=[27,5])
for i in range(20):
    axes[0].plot(t_several[i],x_several[i], linewidth=2)
axes[0].set_xticks([0,20,40,60,80,100])
axes[0].set_xticklabels([0,20,40,60,80,100])
axes[0].set_yticks([-1,0,1])
axes[0].set_yticklabels([-1,0,1])
axes[0].set_xlabel('t')
axes[0].set_ylabel('$X$')
axes[0].set_title('Train datasets(20 timeseries) for Saddle with high noise')

axes[1].plot(t_end_, unif, linewidth=2)
axes[1].plot(t_end_n, unif, '--', linewidth=2)
axes[1].plot(t_end_e, unif, '--', linewidth=2)
axes[1].set_xticks([0,20,40,60,80,100])
axes[1].set_xticklabels([0,20,40,60,80,100])
axes[1].set_yticks([0,0.5,1])
axes[1].set_yticklabels([0,0.5,1])
axes[1].legend(['By 100 Saddle(high noise) timeseries','By 100 simulations from E-MLP with Gaussian noise','By 100 simulations from E-MLP with empirical noise'])
axes[1].set_title('The ECDF of the time that rare event happens')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Probability')

# fig.savefig('./figures/rare_event.eps', dpi=300, bbox_inches='tight')

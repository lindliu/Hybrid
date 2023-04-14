#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:12:21 2023

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append("..") 
from torch import optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import get_dataset, get_cond_noise, get_inter_grid
from model import model_f_hopf, model_fs_dist_hopf
from misc import parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10, lr=5e-4)


model_f_path = params.model_f_path
# model_dist_path = params.model_dist_path
data_pathes = params.data_pathes

data_train_pathes = data_pathes[:params.n_trainset]
x_lower_bound = params.x_lower_bound


model_f = model_f_hopf().to(device)
model_f.load_state_dict(torch.load(model_f_path))

# https://pypi.org/project/MFDFA/
from MFDFA import MFDFA
from MFDFA import fgn

### get data ###
ks = 10
x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
    get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=1, ks=ks, resample=False)

# inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

txs = torch.tensor(np.c_[np.ones_like(s1), x1, s1], dtype=torch.float32).to(device)
numerical = model_f(txs).cpu().detach().numpy()

# s2 = s1+fs*dt+eps, then we can get the empirical distribution of eps under each condition
eps = (s2 - (s1 + numerical[:,[-1]]*dt_s))#/dt_s**.5

### MFDFA analysis
# Select a band of lags, which usually ranges from
# very small segments of data, to very long ones, as
lag = np.unique(np.logspace(0.5, 3, 100).astype(int))

# Obtain the (MF)DFA as
lag, dfa = MFDFA(eps.flatten(), lag = lag, q = 2, order = 1)
plt.loglog(lag, dfa, 'o', label='fOU: MFDFA q=2')
H_hat = np.polyfit(np.log(lag)[4:20],np.log(dfa[4:20]),1)[0]

# Now what you should obtain is: slope = H + 1
print('Estimated H = '+'{:.3f}'.format(H_hat[0])) ###.276

######################################################
######################################################



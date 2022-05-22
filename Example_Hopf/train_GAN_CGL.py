#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:38:36 2022

@author: dliu
"""


# https://github.com/dbgannon/NNets-and-Diffeqns/blob/master/PDEgan_show-full-pinn.ipynb

# This is the GenerativeAdesarialNetwork GAN for solving 
# the non-linear stochastic PDEs.
# actually this example is not full stochastic because it uses the solution
# of a non-linear PDE so we can check output.

#####################
### t is constant ###
#####################
import sys 
sys.path.append("..") 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd


from utils import get_dataset
from model import generator_hopf, discriminator_hopf
from misc import train_model, parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = parameters(data_type='CGL', n_trainset=1, kx=1, ks=10, lr=3e-5)

model_gen_path = params.model_gen_path
model_dis_path = params.model_dis_path
data_pathes = params.data_pathes


data_train_pathes = data_pathes[:params.n_trainset]
### load data ###
x_lower_bound = params.x_lower_bound
kx = params.kx
ks = params.ks
x1, x2, s1, s2, diff_x, diff_s, dt_x, dt_s = \
    get_dataset(data_train_pathes, x_lower_bound=x_lower_bound, kx=kx, ks=ks, resample=False)

plt.figure()
plt.plot(np.sort(dt_x,0))

mask = (dt_x<0.11).flatten()
x1 = x1[mask]
s1 = s1[mask]
x2 = x2[mask]
s2 = s2[mask]
diff_x = diff_x[mask]
dt_x = dt_x[mask]


batch_size = 1024
LAMBDA = .1
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]
def Du(t,u):
    #print(flat(u))
    u_t = autograd.grad(flat(u), t, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    # xv =x[:,0].reshape(batch_size,1)
    # z = u_x*xv
    # u_xx = autograd.grad(flat(z), x, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    # f = u_xx
    f = u_t
    return f


dis = discriminator_hopf().to(device)
gen = generator_hopf().to(device)

# gen.load_state_dict(torch.load(model_gen_path))
# dis.load_state_dict(torch.load(model_dis_path))

true_vals = Variable(torch.ones(batch_size,1)) #torch.ones(batch_size,1)
false_vals = Variable(torch.zeros(batch_size,1)) #torch.zeros(batch_size,1)

num_epochs = 100000
lr = 0.00003
optimizerD = optim.Adam(dis.parameters(), lr=lr)
optimizerG = optim.Adam(gen.parameters(), lr=lr)
one = torch.FloatTensor([1])
minusone = one * -1

ones = torch.ones(batch_size*1, 1).to(device)
num_batches = 100

X = s1
S = x1
S_next = x2
D_s = diff_x
for epoch in range(num_epochs):
    # For each batch
    idx = np.random.choice(np.arange(X.shape[0]), batch_size)
    
    tt_batch = torch.ones(batch_size,1).to(device)
    x1_batch = torch.FloatTensor(X[idx]).to(device)
    s1_batch = torch.FloatTensor(S[idx]).to(device)
    s2_batch = torch.FloatTensor(S_next[idx]).to(device)
    fs_batch = torch.FloatTensor(D_s[idx]).to(device)
    noise = torch.randn(batch_size*1, 2).to(device)

    noisev = torch.cat((tt_batch, x1_batch, s1_batch, noise), 1).to(device)
    noisev.requires_grad=True

    real_data = torch.cat((tt_batch, x1_batch, s1_batch, s2_batch, fs_batch), 1).to(device)
    ###########################
    # (2) Update G network: maximize D(G(z))
    ###########################
    optimizerG.zero_grad()
    fake_s2  = gen(noisev)
    fout1 = Du(noisev, fake_s2[:,0])
    fout2 = Du(noisev, fake_s2[:,1])
    to_dis = torch.cat((tt_batch, x1_batch, s1_batch, fake_s2, fout1[:,[0]], fout2[:,[0]]),1)
    output = dis(to_dis)

    # Calculate G's loss based on this output
    errG = torch.mean(output)

    # Calculate gradients for G
    errG.backward(retain_graph=True)
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()

    ############################
    # (1) Update D network: maximize D(x) - D(G(z))
    ###########################
    ## Train with all-real batch
    optimizerD.zero_grad()
    # Forward pass real batch through D
    real_out = dis(real_data)
    # Calculate loss on all-real batch
    errD_real = torch.mean(real_out)
    # Calculate gradients for D in backward pass
    #errD_real.backward()
    real_out = real_out.mean()
    #real_out.backward(minusone)
    D_x = real_out.item()
    
    ## Train with all-fake batch
    
    fake_s2  = gen(noisev)
    fout1 = Du(noisev, fake_s2[:,0])
    fout2 = Du(noisev, fake_s2[:,1])
    fake_data = torch.cat((tt_batch, x1_batch, s1_batch, fake_s2, fout1[:,[0]], fout2[:,[0]]),1).detach().to(device)
    fake_out = dis(fake_data)
    errD_fake = torch.mean(fake_out)
    fake_out = fake_out.mean()
    D_G_z1 = fake_out.item()
    #errD_fake.backward()

    gradient_penalty = calc_gradient_penalty(dis, real_data, fake_data)
    #gradient_penalty.backward()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real - errD_fake + gradient_penalty
    errD.backward()
    # Update D
    optimizerD.step()

    # Output training stats
    if epoch % 1000 == 0:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f '
                  % (epoch, num_epochs,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2) )
                     #file=open('./pdegan_out.txt','a'))

    # if epoch % 10000 == 0:
    #     #torch.save(dis.state_dict(), 'discriminator'+str(epoch))
    #     torch.save(gen.state_dict(), 'generator-tanh-x2'+str(epoch))


torch.save(gen.state_dict(), model_gen_path)    
torch.save(dis.state_dict(), model_dis_path)    




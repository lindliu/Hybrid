#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:38:28 2022

@author: dliu
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class model_f_saddle(nn.Module):
    def __init__(self):
        
        super(model_f_saddle, self).__init__()
        self.l1 = nn.Linear(3,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50,50)
        self.l4 = nn.Linear(50,50)
        self.l5 = nn.Linear(50,2)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.l5(x)
        return x

class model_f_hopf(nn.Module):
    def __init__(self):
        
        super(model_f_hopf, self).__init__()
        self.l1 = nn.Linear(4,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50,50)
        self.l4 = nn.Linear(50,50)
        self.l5 = nn.Linear(50,3)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.l5(x)
        return x

class model_f_lorenz(nn.Module):
    def __init__(self):
        
        super(model_f_lorenz, self).__init__()
        self.l1 = nn.Linear(4,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50,50)
        self.l4 = nn.Linear(50,50)
        self.l5 = nn.Linear(50,3)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.l5(x)
        return x
    
    
class model_fs_dist_saddle(nn.Module):
    def __init__(self):
        
        super(model_fs_dist_saddle, self).__init__()
        self.l1 = nn.Linear(3,64)
        self.l2 = nn.Linear(64,256)
        self.l3 = nn.Linear(256,256)
        self.l4 = nn.Linear(256,256)
        self.l5 = nn.Linear(256,256)
        self.l6 = nn.Linear(256,256)
        self.l7 = nn.Linear(256,256)
        self.l8 = nn.Linear(256,64)
        self.l9 = nn.Linear(64,1)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.rl(self.l5(x))
        x = self.rl(self.l6(x))
        x = self.rl(self.l7(x))
        x = self.rl(self.l8(x))
        x = self.l9(x)
        return x
    
class model_fs_dist_hopf(nn.Module):
    def __init__(self):
        
        super(model_fs_dist_hopf, self).__init__()
        self.l1 = nn.Linear(4,64)
        self.l2 = nn.Linear(64,256)
        self.l3 = nn.Linear(256,256)
        self.l4 = nn.Linear(256,256)
        self.l5 = nn.Linear(256,256)
        self.l6 = nn.Linear(256,256)
        self.l7 = nn.Linear(256,256)
        self.l8 = nn.Linear(256,64)
        self.l9 = nn.Linear(64,1)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.rl(self.l5(x))
        x = self.rl(self.l6(x))
        x = self.rl(self.l7(x))
        x = self.rl(self.l8(x))
        x = self.l9(x)
        return x


class model_std_mean_saddle(nn.Module):
    def __init__(self):
        
        super(model_std_mean_saddle, self).__init__()
        self.l1 = nn.Linear(3,32)
        self.l2 = nn.Linear(32,128)
        self.l3 = nn.Linear(128,128)
        self.l4 = nn.Linear(128,128)
        self.l5 = nn.Linear(128,128)
        self.l6 = nn.Linear(128,64)
        self.l7 = nn.Linear(64,2)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.rl(self.l5(x))
        x = self.rl(self.l6(x))
        x = self.l7(x)
        return x
    
    

class model_std_mean_hopf(nn.Module):
    def __init__(self):
        
        super(model_std_mean_hopf, self).__init__()
        self.l1 = nn.Linear(4,32)
        self.l2 = nn.Linear(32,128)
        self.l3 = nn.Linear(128,128)
        self.l4 = nn.Linear(128,128)
        self.l5 = nn.Linear(128,128)
        self.l6 = nn.Linear(128,64)
        self.l7 = nn.Linear(64,2)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        x = self.rl(self.l1(x))
        x = self.rl(self.l2(x))
        x = self.rl(self.l3(x))
        x = self.rl(self.l4(x))
        x = self.rl(self.l5(x))
        x = self.rl(self.l6(x))
        x = self.l7(x)
        return x
    
def f(t, net, flag=0):
    t = torch.autograd.Variable(t, requires_grad=True)
    u = net(t) # the dependent variable u is given by the network based on independent variables x,t
    # u_t = torch.autograd.grad(u, t, create_graph=True)[0]
    u_t = torch.autograd.grad(outputs=u, inputs=t, 
        grad_outputs=torch.ones(u.size()).to(device),create_graph=True, retain_graph=True)[0]

    u_t = torch.autograd.grad(outputs=u, inputs=t,
                              grad_outputs=torch.ones(u.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    return u_t

    
# defining generator class
class generator(nn.Module):
    
    def __init__(self):
        
        super(generator, self).__init__()
        self.l1 = nn.Linear(5,256)  #input [x, v], v is random variable
        self.l2 = nn.Linear(256,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,800)
        self.l5 = nn.Linear(800,512)
        self.l6 = nn.Linear(512,512)
        self.l7 = nn.Linear(512,64) #output is generated sigma
        self.l8 = nn.Linear(64,1) #output is generated sigma
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        u = self.rl(self.l4(u))
        u = self.rl(self.l5(u))
        u = self.rl(self.l6(u))
        u = self.rl(self.l7(u))
        z = self.l8(u)
        return z

class discriminator(nn.Module):
    
    def __init__(self):
        
        super(discriminator, self).__init__()
        self.l1 = nn.Linear(5,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.l4  = nn.Linear(256,50)
        self.l5  = nn.Linear(50,10)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        u = self.l4(u)
        out = self.l5(u)
 
        return out




# defining generator class
class generator_hopf(nn.Module):
    
    def __init__(self):
        
        super(generator_hopf, self).__init__()
        self.l1 = nn.Linear(6,256)  #input [x, v], v is random variable
        self.l2 = nn.Linear(256,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,800)
        self.l5 = nn.Linear(800,512)
        self.l6 = nn.Linear(512,512)
        self.l7 = nn.Linear(512,64) #output is generated sigma
        self.l8 = nn.Linear(64,2) #output is generated sigma
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        u = self.rl(self.l4(u))
        u = self.rl(self.l5(u))
        u = self.rl(self.l6(u))
        u = self.rl(self.l7(u))
        z = self.l8(u)
        return z

class discriminator_hopf(nn.Module):
    
    def __init__(self):
        
        super(discriminator_hopf, self).__init__()
        self.l1 = nn.Linear(8,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.l4  = nn.Linear(256,50)
        self.l5  = nn.Linear(50,10)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        u = self.l4(u)
        out = self.l5(u)
 
        return out




# defining generator class
class generator_hopf1(nn.Module):
    
    def __init__(self):
        
        super(generator_hopf1, self).__init__()
        self.l1 = nn.Linear(6,256)  #input [x, v], v is random variable
        self.l2 = nn.Linear(256,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,800)
        self.l5 = nn.Linear(800,512)
        self.l6 = nn.Linear(512,512)
        self.l7 = nn.Linear(512,64) #output is generated sigma
        self.l8 = nn.Linear(64,3) #output is generated sigma
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        u = self.rl(self.l4(u))
        u = self.rl(self.l5(u))
        u = self.rl(self.l6(u))
        u = self.rl(self.l7(u))
        z = self.l8(u)
        return z
    
class discriminator_hopf1(nn.Module):
    
    def __init__(self):
        
        super(discriminator_hopf1, self).__init__()
        self.l1 = nn.Linear(10,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.l4  = nn.Linear(256,50)
        self.l5  = nn.Linear(50,10)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        u = self.l4(u)
        out = self.l5(u)
 
        return out




# defining generator class
class generator_lorenz(nn.Module):
    
    def __init__(self):
        
        super(generator_lorenz, self).__init__()
        self.l1 = nn.Linear(6,256)  #input [x, v], v is random variable
        self.l2 = nn.Linear(256,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,800)
        self.l5 = nn.Linear(800,512)
        self.l6 = nn.Linear(512,512)
        self.l7 = nn.Linear(512,64) #output is generated sigma
        self.l8 = nn.Linear(64,3) #output is generated sigma
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        u = self.rl(self.l4(u))
        u = self.rl(self.l5(u))
        u = self.rl(self.l6(u))
        u = self.rl(self.l7(u))
        z = self.l8(u)
        return z

class discriminator_lorenz(nn.Module):
    
    def __init__(self):
        
        super(discriminator_lorenz, self).__init__()
        self.l1 = nn.Linear(10,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.l4  = nn.Linear(256,50)
        self.l5  = nn.Linear(50,10)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        u = self.l4(u)
        out = self.l5(u)
 
        return out


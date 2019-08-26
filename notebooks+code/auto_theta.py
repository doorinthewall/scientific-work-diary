import pickle as pk
import algorithms as algo

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.special import logsumexp
import pandas as pd
import pandasql as ps
from copy import deepcopy
from scipy.optimize import minimize 
from tqdm import tqdm_notebook
from torch import nn
import torch
from torch.distributions import Categorical, Normal



def envelope_gp_cnn_theta(sources, data, target, target_points_to_start, bounds, loss,\
             search_grid, n_restarts=10, number_of_iterations=100, sigma_msr=1e-10, forget=100,\
         forget_upd=0.8, tao=1e-2, nu=1, gamma=None,\
                       lr_bandit_weights=None, strategy = 'exp3'):
    

    forget_theta_reward_coef = 0.8
    net_bias = 0
    forget_grid = 10.**np.arange(-2,2)
    policy_shape = forget_grid.shape
    forget_grid = forget_grid.ravel()
    net_dim = forget_grid.shape[0]
    net = nn.Sequential(nn.Linear(net_dim,net_dim), nn.Tanh())
    net_const_ = torch.Tensor([1]*forget_grid.shape[0])[np.newaxis, np.newaxis, :]
    net_h_ = None
    forget_= 0
    optimizer = torch.optim.Adam(net.parameters(), lr=0.5*1e-2)

    target_data = target_points_to_start.copy()
    
    
    #envelope_gp
    sigma_s = [sigma_msr] + [nu/(tao+1)]*len(sources)   # добавим пустой источник
    sources_ = [lambda data: None] + sources # добавим пустой источник
    gp = GaussianProcessRegressor()
    gps = GaussianProcessRegressor()
    
    #MAB
    log_weights = np.array([0.0] * len(sources_))
    history = []
    dim = data.shape[1]
    K = len(sources_)
    
    #theorem 1 exp3_ix
    if gamma is None:
        if strategy == 'exp3-IX':
            gamma = np.sqrt(2*np.log(K)/(K*number_of_iterations))
        if strategy == 'exp3-auer':
            gamma = 0.9
        if strategy == 'exp3':
            gamma = 0
        
    if lr_bandit_weights is None:
        if strategy == 'exp3-IX':
            lr_bandit_weights = 2*gamma
        if strategy == 'exp3-auer':
            lr_bandit_weights = gamma
        if strategy == 'exp3':
            lr_bandit_weights = np.sqrt(2*np.log(K)/(K*number_of_iterations))
        
        
    #main
    for _ in tqdm_notebook(range(number_of_iterations), leave=False):
        
        #distribution update
        corrected_weights = log_weights - np.max(log_weights)
        theSum = logsumexp(corrected_weights)
        if strategy == 'exp3-auer':
            probabilityDistribution = \
            (1.0 - gamma) * np.exp(corrected_weights - theSum) + (gamma / log_weights.shape[0])
        elif strategy == 'exp3-IX' or strategy == 'exp3':
            probabilityDistribution = np.exp(corrected_weights - theSum)
        
        #draw
        arm = algo.draw(probabilityDistribution)
        
        
        alpha=np.vstack((sigma_s[arm]*np.ones((data.shape[0],1)),\
                                       sigma_msr*np.ones((target_data.shape[0],1)))).ravel()
        
        if arm != 0:
            gp.set_params(alpha=alpha)
            y = np.vstack((sources_[arm](data)[:,np.newaxis], target(target_data)[:,np.newaxis])).ravel()
            gp.fit(np.vstack((data, target_data)), y)
            gps.fit(data, sources_[arm](data))
        else:
            gp.set_params(alpha=sigma_msr)
            y = target(target_data)
            gp.fit(target_data, y)
            gps = deepcopy(gp)
           
        policy = nn.Softmax(0)(net(net_const_).reshape(-1)).reshape(-1)
              
        m = Categorical(policy)
        forget_ = m.sample()
        log_policy = m.log_prob(forget_)
        forget_ = forget_grid[forget_.tolist()]
            
        
        
        expected_improvement = algo.get_gp_ucb_simple_sklearn(search_grid, gp, forget_)
        
        min_val= -np.max(expected_improvement)
        new_point = search_grid[np.argmax(expected_improvement)]
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            min_obj = lambda x: -algo.get_gp_ucb_simple_sklearn([x], gp, forget_)
            res = minimize(min_obj, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < min_val:
                min_val = res.fun
                new_point = res.x
        
        
        theReward = -loss(gps.predict(new_point[np.newaxis,:]),target(new_point))**2
        
        #net to control theta optimize step        
        net_reward = -(1 - target(new_point))**2
        if isinstance(net_reward, np.ndarray):
            net_reward = net_reward[0]
        
        net_loss = (net_reward - net_bias) * (-log_policy)
        net_bias = (net_bias+net_reward)*forget_theta_reward_coef
                
        optimizer.zero_grad()
        net_loss.backward()
        optimizer.step()
        
        history += [(arm, probabilityDistribution, theReward, policy.reshape(policy_shape), net_reward, net_bias, forget_)]
            
        log_weights[arm] += theReward * lr_bandit_weights / \
                        (probabilityDistribution[arm] + gamma)
        
        target_data = np.vstack((target_data, np.array([new_point])))
        tao += 0.5
        nu += ((target(new_point) - gps.predict(new_point[np.newaxis,:]))**2)/2
        sigma_s[arm] = nu/(tao + 1)
      
        
    return  target_data, history, gp





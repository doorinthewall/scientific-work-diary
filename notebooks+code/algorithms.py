import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.special import logsumexp
from copy import deepcopy
from scipy.optimize import minimize 
from tqdm import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder



def get_gp_ucb_simple_sklearn( X, gp, forget=100):
    mu, sigma = gp.predict(X, True)
    return mu + sigma * np.sqrt(forget) 

def get_gp_ucb_advanced_sklearn( mu, sigma, delta, iteration, D, forget):
    rdelta = np.min(np.max(0,delta), 1)
    forget = 2*(2*np.log(iteration) + 2*np.log(np.pi)\
    - np.log(rdelta) + np.log(6) + np.log(D))
    return mu + sigma * np.sqrt(forget)

def draw(weights):
    return weights.argmax()

def myLoss1(mu,f):
    return (f-mu)**2
    
def myLoss2(mu,f):
    return (f-1)**2


def envelope_gp(sources, data, target, target_points_to_start, bounds, loss, search_grid, n_restarts=10,\
                    number_of_iterations=100, sigma_msr=1e-10, forget=100, forget_upd=0.8, tao=1e-2, nu=1, gamma=None,\
                       lr_bandit_weights=None, strategy = 'exp3'):
    
    forget_= forget

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
    for _ in tqdm_notebook(range(number_of_iterations)):
        
        #distribution update
        corrected_weights = log_weights - np.max(log_weights)
        theSum = logsumexp(corrected_weights)
        if strategy == 'exp3-auer':
            probabilityDistribution = \
            (1.0 - gamma) * np.exp(corrected_weights - theSum) + (gamma / log_weights.shape[0])
        elif strategy == 'exp3-IX' or strategy == 'exp3':
            probabilityDistribution = np.exp(corrected_weights - theSum)
        
        #draw
        arm = draw(probabilityDistribution)
#        history += [arm]
        
        
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
        
        expected_improvement = get_gp_ucb_simple_sklearn(search_grid, gp, forget_)
        
        min_val= -np.max(expected_improvement)
        new_point = search_grid[np.argmax(expected_improvement)]
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            min_obj = lambda x: -get_gp_ucb_simple_sklearn([x], gp, forget_)
            res = minimize(min_obj, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < min_val:
                min_val = res.fun
                new_point = res.x
        
        #new_point = search_grid[np.argmax(expected_improvement)]
        
        theReward = -loss(gps.predict(new_point[np.newaxis,:]),target(new_point))**2
            
        
        history += [(arm, probabilityDistribution, theReward)]
            
        log_weights[arm] += theReward * lr_bandit_weights / \
                        (probabilityDistribution[arm] + gamma)
        
        target_data = np.vstack((target_data, np.array([new_point])))
        tao += 0.5
        nu += ((target(new_point) - gps.predict(new_point[np.newaxis,:]))**2)/2
        sigma_s[arm] = nu/(tao + 1)
        
        forget_ *= forget_upd
        
    return  target_data, history, gp, 





def SMBO_transfer(sources, data, target, target_points_to_start, bounds, search_grid, n_restarts=10,\
                    number_of_iterations=100, forget=100):
    
    target_data = target_points_to_start.copy() 
    t = target(target_points_to_start)
#    target_data = np.append(np.zeros((target_data.shape[0], 1)),\
#              target_data, axis=1) #добавим признак-индекс датасета, целевой == 0

    
    
    mu = t.mean()
    sigma = ((t - mu)**2).sum()**0.5
    
    enc = OneHotEncoder()

    source_marks = enc.fit_transform(np.arange(len(sources)+1).reshape(-1,1)).toarray()
    

    target_data = np.append(source_marks[0,:][np.newaxis,:] + np.zeros((target_data.shape[0], 1)),\
              target_data, axis=1)
    
    search_grid_ = np.append(source_marks[0,:][np.newaxis,:] + np.zeros((search_grid.shape[0], 1)),\
                             search_grid, axis=1)
    
    X = []
    y = []
    for i,s in enumerate(sources):
        f = s(data)
        mu_ = f.mean()
        sigma_ = ((f - mu_)**2).sum()**0.5
        y += [(f - mu_)/sigma_]
        data_ = data.copy()
#        data_ = np.append((i+1)*np.ones((data.shape[0], 1)), data, axis=1)
        data_ = np.append(source_marks[i+1,:][np.newaxis,:] + np.zeros((data.shape[0],1))\
                          , data, axis=1)
        X += [data_]
        
    y = np.hstack(y)
    X = np.vstack(X)

    gp = GaussianProcessRegressor()
    
    dim = data.shape[1]
    
    for _ in tqdm_notebook(range(number_of_iterations)):
        
        t_norm = (t - mu)/sigma
        #print(target_data.shape, target(target_data).shape)
        gp.fit(np.vstack((X, target_data)), np.hstack((y, t_norm)))
        
        expected_improvement = get_gp_ucb_simple_sklearn(search_grid_, gp, forget)
        
        min_val= -np.max(expected_improvement)
        new_point = search_grid_[np.argmax(expected_improvement)]
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            min_obj = lambda x: -get_gp_ucb_simple_sklearn([x], gp, forget)
            x0 = np.append(source_marks[0,:], x0)
            bounds_ = np.append(source_marks[0,:][:, np.newaxis] + np.zeros((1,bounds.shape[1]))\
                                , bounds, axis=0)
            res = minimize(min_obj, x0, method='L-BFGS-B', bounds=bounds_)
            if res.fun < min_val:
                min_val = res.fun
                new_point = res.x
                
        t = np.append(t, target(new_point[source_marks.shape[1]:]))
        target_data = np.append(target_data, new_point[np.newaxis,:], axis=0)
        
        mu = t.mean()
        sigma = ((t - mu)**2).sum()**0.5
        
    
    return target_data[:,source_marks.shape[1]:], gp, source_marks
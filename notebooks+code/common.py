import numpy as np  

def average_max(target_data, f):
    res = np.array([f(obj) for obj in target_data])

    curr_max = res[:,0].copy()
    for i in range(1,res.shape[1]):
        mask = res[:,i] > curr_max
        curr_max[mask] = res[:,i][mask]
        res[:,i] = curr_max.copy()
        
    sigma = ((res - res.mean(axis=0))**2).sum()**0.5

    return res.mean(axis=0)
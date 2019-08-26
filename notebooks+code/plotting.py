from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from IPython.core.debugger import set_trace

def plot_sources(sources, f, X, Y, data):
    fig = plt.figure(figsize=(15,10))
    for i, s in enumerate(sources):
        ax = fig.add_subplot(2, len(sources)/2, i+1, projection='3d')
        ax.view_init(elev=40., azim=110)
        ax.set_title('source #{}'.format(i+1))
        Z = f(data)
        ax.plot_wireframe(X, Y,
                Z.reshape(X.shape), alpha=0.5, color='g', label='target')
        Z = s(data)
        ax.plot_wireframe(X, Y,
                Z.reshape(X.shape), alpha=0.5, color='b', label='source')
        ax.legend()
    return fig

def count_arms(histories, sources_num, grid=5):
    hists = {}
    arms = np.arange(sources_num+1)
    
    fig = plt.figure()
    
    for history in histories:
        history = np.array([event[0] for event in history])
        for i in arms:
            hists[i] = hists.get(i,0) + np.cumsum(history == i)


    bottom=0
    res = []
    for i, label in zip(arms, ['no source'] + ['source #{}'.format(j+1) for j in range(sources_num)]):
        aid = []
        for j in np.arange(len(history))[::grid]:
            aid += hists[i][j] * [j]
        res += [aid]
    
    plt.hist(res, histtype='bar', label=['no source']\
             + ['source #{}'.format(j+1) for j in range(sources_num)])
    
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('count')
    
    return fig

def plot_exp_res(labels, data):
    fig = plt.figure()
    for l, d in zip(labels, data):
        plt.plot(np.arange(d.shape[0]), d, 'o--', label=l)
    plt.legend()
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('max value achieved')
    return fig

def plot_res_func(f, X, Y, data, gps, source_marks=None):
    
    fig = plt.figure(figsize=(7,5))
    ax = Axes3D(fig)
    ax.view_init(elev=40., azim=130)
    Z = f(data)
    ax.plot_wireframe(X, Y,
            Z.reshape(X.shape), alpha=0.5, color='g', label='target')
    
    if source_marks is not None:
        data_ = np.append(source_marks[0,:][np.newaxis,:] + np.zeros((data.shape[0],1))\
                          , data, axis=1)
    else:
        data_ = data.copy()

    Z = np.array([gp.predict(data_) for gp in gps])    
    ax.plot_wireframe(X, Y, Z.mean(axis=0).reshape(X.shape), alpha=0.5, color='b', label='approx')
    ax.legend()
    return fig, ax
""" 
Demonstration of the universal approximation capability of an NN.
Built using `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../nabla_python")
from nabla import Tensor


def sample_data():
    
    # Train samples
    xtrain = (np.random.rand(1, 1024) - 0.5) * 2 * np.pi
    noise = np.random.normal(0, 0.01, size=(1, 1024))
    ytrain = np.sin(xtrain) + noise

    # Test samples
    xtest = np.expand_dims(np.linspace(-np.pi, np.pi, 1024), axis=0)
    ytest = np.sin(xtest)

    # Dummy value "1" to use with learnable bias
    xtrain = np.concatenate((xtrain, np.ones_like(xtrain)), axis=0)
    xtest = np.concatenate((xtest, np.ones_like(xtest)), axis=0)

    # To Tensor
    xtrain, ytrain, xtest, ytest = Tensor(xtrain), Tensor(ytrain), Tensor(xtest), Tensor(ytest)

    return xtrain, ytrain, xtest, ytest

def model(x, params):
    z1 = params['w1'].dot(x)
    a1 = z1.sigmoid()
    z2 = params['w2'].dot(a1)
    y = z2.tanh()          
    return y

def init_model_params():
    params = {
    'w1': Tensor(np.random.normal(size=(16, 2)), requires_grad=True),
    'w2': Tensor(np.random.normal(size=(1, 16)), requires_grad=True)
    }
    return params

def update_params(params, lr):
    for k in params.keys():
        params[k].data = params[k].data - lr*params[k].grad
    return params

def zero_grad(params):
    for k in params.keys():
        params[k].grad = np.zeros_like(params[k].data)
    return params

def mse_loss(pred, gt):
    loss = (pred - gt) ** Tensor(np.array(2.))
    loss = loss.sum() / Tensor(np.array(gt.shape[1]))
    return loss

def run_demo():
    
    # Init model
    params = init_model_params()

    # Init viz
    fig, ax = plt.subplots()
    xtrain, ytrain, xtest, ytest = sample_data()    
    ytestpred = model(xtest, params)
    testgt_plot = plt.plot(xtest.data[0,:], ytest.data[0,:], c='tab:blue')[0]
    testpred_plot = plt.plot(xtest.data[0,:], ytestpred.data[0,:], c='tab:orange')[0]
    plt.ion(); plt.show()    

    # Training loop
    for epoch in tqdm(range(1, 1000)):

        # Update model
        xtrain, ytrain, xtest, ytest = sample_data()
        ypred = model(xtrain, params)
        loss = mse_loss(ypred, ytrain)
        loss.backward()
        params = update_params(params, lr=1e-1)
        params = zero_grad(params)

        # Test and viz
        ytestpred = model(xtest, params)
        testpred_plot.set_ydata(ytestpred.data[0,:])
        fig.canvas.draw()
        fig.canvas.flush_events()    


if __name__ == '__main__':
    run_demo()
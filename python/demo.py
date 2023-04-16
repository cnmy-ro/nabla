""" 
Demonstration of the universal approximation capability of an NN.
Built using `gradcore`.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from gradcore import Tensor, Operator



def generate_data():
    
    xtrain = (np.random.rand(1000) - 0.5) * 2 * np.pi
    noise = np.random.normal(0, 0.01, size=(1000,))
    ytrain = np.sin(xtrain) + noise

    xtest = np.linspace(-np.pi, np.pi, 1000)
    ytest = np.sin(xtest)

    # Dummy value "1" to use with learnable bias
    xtrain = np.stack((xtrain, np.ones_like(xtrain)), axis=1)
    xtest = np.stack((xtest, np.ones_like(xtest)), axis=1)

    # Add extra dims to xtrain, ytrain, xtest since gradcore requires 2D arrays
    xtrain = np.expand_dims(xtrain, axis=2)
    xtest = np.expand_dims(xtest, axis=2)
    ytrain = np.expand_dims(np.expand_dims(ytrain, axis=1), axis=2)

    return xtrain, ytrain, xtest, ytest

def model(x, params):
    z1 = params['w1'].dot(x)
    a1 = z1.sigmoid()
    z2 = params['w2'].dot(a1)
    y = z2.tanh()          
    return y

def init_params():
    params = {
    'w1': Tensor(np.random.normal(size=(8, 2))),
    'w2': Tensor(np.random.normal(size=(1, 8)))
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

def infer(params, xtest):
    ytestpred = []
    for i in range(xtest.shape[0]):        
        xtest_i = Tensor(xtest[i])
        ypred_i = model(xtest_i, params)
        ytestpred.append(np.squeeze(ypred_i.data))
    ytestpred = np.array(ytestpred)
    return ytestpred


def run_demo():
    
    xtrain, ytrain, xtest, ytest = generate_data()

    params = init_params()
    ytestpred = infer(params, xtest)
    
    fig, ax = plt.subplots()
    testgt_plot = plt.plot(xtest[:, 0].squeeze(), ytest, c='tab:blue')[0]
    testpred_plot = plt.plot(xtest[:, 0].squeeze(), ytestpred, c='tab:orange')[0]
    plt.ion()
    plt.show()

    for epoch in tqdm(range(1, 50)):

        for i in range(xtrain.shape[0]):
            
            xtrain_i = Tensor(xtrain[i])
            ytrain_i = Tensor(ytrain[i])
            ypred_i = model(xtrain_i, params)

            loss = (ypred_i - ytrain_i) ** Tensor(np.array([[2.]]), requires_grad=False)
            loss.backward()
            params = update_params(params, lr=0.5)
            params = zero_grad(params)                        

        ytestpred = infer(params, xtest)
        testpred_plot.set_ydata(ytestpred)
        fig.canvas.draw()
        time.sleep(0.005)
        fig.canvas.flush_events()    



if __name__ == '__main__':
    run_demo()
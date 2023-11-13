""" 
Regression model built with `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../nabla_python")
from nabla import Tensor


# ---
# Utils

def sample_data():
    
    # Train samples
    xtrain = (np.random.rand(1, 1024) - 0.5) * 2 * np.pi
    noise = np.random.normal(0, 0.01, size=(1, 1024))
    ytrain = np.sin(xtrain) + noise

    # Test samples
    xtest = np.expand_dims(np.linspace(-np.pi, np.pi, 1024), axis=0)
    ytest = np.sin(xtest)

    # To Tensor, shape (len, batch)
    xtrain, ytrain, xtest, ytest = Tensor(xtrain), Tensor(ytrain), Tensor(xtest), Tensor(ytest)

    return xtrain, ytrain, xtest, ytest

class MLP:
    def __init__(self):
        self.params = {
        'w1': Tensor(np.random.normal(size=(16, 1)), requires_grad=True),
        'b1': Tensor(np.random.normal(size=(16, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(1, 16)), requires_grad=True),
        'b2': Tensor(np.random.normal(size=(1, 1)), requires_grad=True),
        }
    def __call__(self, x):
        a1 = (self.params['w1'].dot(x) + self.params['b1']).sigmoid()
        y = (self.params['w2'].dot(a1) + self.params['b2']).tanh()
        return y

def update_params(model, lr):
    for param in model.params.values():
        param.data = param.data - lr*param.grad
    return model

def zero_grad(model):
    for param in model.params.values():
        param.grad = np.zeros_like(param.data)
        param.prev = None
    return model

def mse_loss(pred, gt):
    loss = ((pred - gt)**2).mean()
    return loss


# ---
# Main function

def main():
    
    # Init model
    model = MLP()

    # Init viz
    fig, ax = plt.subplots()
    xtrain, ytrain, xtest, ytest = sample_data()    
    ytestpred = model(xtest)
    testgt_plot = plt.plot(xtest.data[0,:], ytest.data[0,:], c='tab:blue')[0]
    testpred_plot = plt.plot(xtest.data[0,:], ytestpred.data[0,:], c='tab:orange')[0]
    plt.ion(); plt.show()    

    # Training loop
    for epoch in tqdm(range(1000)):

        # Update model
        xtrain, ytrain, xtest, ytest = sample_data()
        ypred = model(xtrain)
        loss = mse_loss(ypred, ytrain)
        loss.backward()
        model = update_params(model, lr=1e-1)
        model = zero_grad(model)

        # Test and viz
        ytestpred = model(xtest)
        model = zero_grad(model)
        testpred_plot.set_ydata(ytestpred.data[0,:])
        fig.canvas.draw()
        fig.canvas.flush_events()    


# ---
# Run

if __name__ == '__main__':
    main()
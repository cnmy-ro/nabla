""" 
Regression model built with `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../python")
import nabla
from nabla import Tensor
from utils import AdamOptimizer, zero_grad


# ---
# Config
BATCH_SIZE = 512
ITERS = 500


# ---
# Utils

class MLP:
    def __init__(self):
        self.params = {
        'w1': nabla.randn((1, 128), requires_grad=True), 'b1': nabla.zeros((1, 128), requires_grad=True),
        'w2': nabla.randn((128, 1), requires_grad=True), 'b2': nabla.zeros((1, 1), requires_grad=True),
        }
    def __call__(self, x):
        a1 = (x.dot(self.params['w1']) + self.params['b1']).sigmoid()
        y = a1.dot(self.params['w2']) + self.params['b2']
        return y

def sample_data():
    
    # Train samples
    xtrain = (np.random.rand(BATCH_SIZE, 1) - 0.5) * 2 * np.pi
    noise = np.random.normal(0, 0.1, size=(BATCH_SIZE, 1))
    ytrain = np.sin(xtrain) + noise

    # Test samples
    xtest = np.expand_dims(np.linspace(-np.pi, np.pi, BATCH_SIZE), axis=1)
    ytest = np.sin(xtest)

    # To Tensor, shape (len, batch)
    xtrain, ytrain, xtest, ytest = Tensor(xtrain), Tensor(ytrain), Tensor(xtest), Tensor(ytest)

    return xtrain, ytrain, xtest, ytest

def mse_loss(pred, gt):
    loss = ((pred - gt)**2).mean()
    return loss


# ---
# Main function

def main():
    
    # Init model and opt
    model = MLP()
    opt = AdamOptimizer(model, lr=1e-2)

    # Init viz
    fig = plt.figure()
    xtrain, ytrain, xtest, ytest = sample_data()
    nabla.enable_grad(False)
    ytestpred = model(xtest)
    nabla.enable_grad(True)
    testgt_plot = plt.plot(xtrain.data[:,0], ytrain.data[:,0], c='tab:blue', marker='.', ls='', alpha=0.5)[0]
    testpred_plot = plt.plot(xtest.data[:,0], ytestpred.data[:,0], c='tab:orange')[0]
    plt.ylim(-1.2, 1.2)
    plt.ion(); plt.show()    

    # Training loop
    for it in tqdm(range(ITERS)):

        # Update model
        xtrain, ytrain, xtest, ytest = sample_data()
        ypred = model(xtrain)
        loss = mse_loss(ypred, ytrain)
        loss.backward()
        model = opt.step(model)
        model = zero_grad(model)
        
        # Test and viz
        nabla.enable_grad(False)
        ytestpred = model(xtest)
        nabla.enable_grad(True)
        testpred_plot.set_ydata(ytestpred.data[:,0])
        fig.canvas.draw()
        fig.canvas.flush_events()
        # fig.savefig(f"./outputs/regression/{str(it).zfill(5)}.png")


# ---
# Run
if __name__ == '__main__':
    main()
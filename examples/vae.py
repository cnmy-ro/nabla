""" 
Variational autoencoder built with `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm

sys.path.append("../nabla_python")
from nabla import Tensor
from utils import AdamOptimizer


# ---
# Constants
E = Tensor(np.array(2.718))


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 256
BATCH_SIZE = 64
ITERS = 1000
LR = 1e-3
BETA = 5e2


# ---
# Utils

class Encoder:
    def __init__(self):
        self.params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, 2)), requires_grad=True),                 'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),        'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),        'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4_mean': Tensor(np.random.normal(size=(LATENT_DIM, HIDDEN_DIM)), requires_grad=True),   'b4_mean': Tensor(np.random.normal(size=(LATENT_DIM, 1)), requires_grad=True),
        'w4_logvar': Tensor(np.random.normal(size=(LATENT_DIM, HIDDEN_DIM)), requires_grad=True), 'b4_logvar': Tensor(np.random.normal(size=(LATENT_DIM, 1)), requires_grad=True)
        }
    def __call__(self, x):
        a1 = (self.params['w1'].dot(x) + self.params['b1']).sigmoid()
        a2 = (self.params['w2'].dot(a1) + self.params['b2']).sigmoid()
        a3 = (self.params['w3'].dot(a2) + self.params['b3']).sigmoid()
        z_mean = self.params['w4_mean'].dot(a3) + self.params['b4_mean']
        z_logvar = self.params['w4_logvar'].dot(a3) + self.params['b4_logvar']
        return z_mean, z_logvar

class Decoder:
    def __init__(self):
        self.params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, LATENT_DIM)), requires_grad=True), 'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4': Tensor(np.random.normal(size=(2, HIDDEN_DIM)), requires_grad=True),          'b4': Tensor(np.random.normal(size=(2, 1)), requires_grad=True)
        }
    def __call__(self, z):
        a1 = (self.params['w1'].dot(z) + self.params['b1']).sigmoid()
        a2 = (self.params['w2'].dot(a1) + self.params['b2']).sigmoid()
        a3 = (self.params['w3'].dot(a2) + self.params['b3']).sigmoid()
        x = self.params['w4'].dot(a3) + self.params['b4']
        return x
        
def sample_data():
    """
    Data generation process. Unknown to the model.
    """
    X, _ = datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.1)
    data_sample = np.stack([X[:, 0], X[:, 2]], axis=1)
    data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
    data_sample = data_sample * 2. - 1.
    data_sample = data_sample.T  # To shape (len, batch)
    data_sample = Tensor(data_sample)
    return data_sample

def sample_model(dec):
    z = Tensor(np.random.randn(LATENT_DIM, BATCH_SIZE))
    model_sample = dec(z)
    dec = zero_grad(dec)
    return model_sample

def encode_decode(enc, dec, data_sample):
    z_mean, z_logvar = enc(data_sample)
    z = z_mean + E**(z_logvar * 0.5) * Tensor(np.random.randn(LATENT_DIM, BATCH_SIZE))  # Reparam trick
    recon = dec(z)
    return z_mean, z_logvar, recon

def mse_loss(pred, gt):
    loss = ((pred - gt)**2).mean()
    return loss

def kl_loss(z_mean, z_logvar):
    loss = (-z_mean**2 - E**z_logvar + 1).mean() * (-0.5)
    return loss

def zero_grad(model):
    for param in model.params.values():
        param.grad = np.zeros_like(param.grad)
        param.prev = None
    return model


# ---
# Main function

def main():

    # Init model and opts
    enc = Encoder()
    dec = Decoder()
    opt_e = AdamOptimizer(enc, lr=LR)
    opt_d = AdamOptimizer(dec, lr=LR)

    # Init viz
    losses_input_recon, losses_latent_prior = [], []
    data_sample = sample_data()
    model_sample = sample_model(dec)
    fig, ax = plt.subplots()
    data_sample_plot = ax.plot(data_sample.data[0, :], data_sample.data[1, :], c='tab:blue', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample.data[0, :], model_sample.data[1, :], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(ITERS)):

        # Update model
        data_sample = sample_data()
        latents_mean, latents_logvar, recon = encode_decode(enc, dec, data_sample)
        loss_input_recon = mse_loss(recon, data_sample)
        loss_latent_prior = kl_loss(latents_mean, latents_logvar)
        loss = loss_input_recon + loss_latent_prior * BETA
        loss.backward()
        enc = opt_e.step(enc)
        dec = opt_d.step(dec)

        # Sample and viz
        losses_input_recon.append(loss_input_recon.data.squeeze()); losses_latent_prior.append(loss_latent_prior.data.squeeze())
        if it % 10 == 0:
            data_sample_plot.set_xdata(data_sample.data[0, :]); data_sample_plot.set_ydata(data_sample.data[1, :])
            model_sample = sample_model(dec)
            model_sample_plot.set_xdata(model_sample.data[0, :]); model_sample_plot.set_ydata(model_sample.data[1, :])
            fig.canvas.draw(); fig.canvas.flush_events()

    # Plot loss curves
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses_input_recon, label='Recon'); ax.plot(losses_latent_prior, label='Prior')
    ax.set_title("Training loss"); ax.legend(); plt.show()


# ---
# Run
if __name__ == '__main__':
    main()
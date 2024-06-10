""" 
Variational autoencoder built with `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm

sys.path.append("../pynabla")
import nabla
from nabla import Tensor
from utils import AdamOptimizer, zero_grad


# ---
# Constants
E = Tensor(np.array(2.718))


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 256
BATCH_SIZE = 256
ITERS = 1000
LR = 1e-3
BETA = 1e-2


# ---
# Utils

class Encoder:
    def __init__(self):
        self.params = {
        'w1': nabla.randn((2, HIDDEN_DIM), requires_grad=True),                 'b1': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w2': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True),        'b2': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w3': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True),        'b3': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w4_mean': nabla.randn((HIDDEN_DIM, LATENT_DIM), requires_grad=True),   'b4_mean': nabla.zeros((1, LATENT_DIM), requires_grad=True),
        'w4_logvar': nabla.randn((HIDDEN_DIM, LATENT_DIM), requires_grad=True), 'b4_logvar': nabla.zeros((1, LATENT_DIM), requires_grad=True)
        }
    def __call__(self, x):
        a1 = (x.dot(self.params['w1']) + self.params['b1']).sigmoid()
        a2 = (a1.dot(self.params['w2']) + self.params['b2']).sigmoid()
        a3 = (a2.dot(self.params['w3']) + self.params['b3']).sigmoid()
        z_mean = a3.dot(self.params['w4_mean']) + self.params['b4_mean']
        z_logvar = a3.dot(self.params['w4_logvar']) + self.params['b4_logvar']
        return z_mean, z_logvar

class Decoder:
    def __init__(self):
        self.params = {
        'w1': nabla.randn((LATENT_DIM, HIDDEN_DIM), requires_grad=True), 'b1': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w2': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True), 'b2': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w3': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True), 'b3': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w4': nabla.randn((HIDDEN_DIM, 2), requires_grad=True),          'b4': nabla.zeros((1, 2), requires_grad=True)
        }
    def __call__(self, z):
        a1 = (z.dot(self.params['w1']) + self.params['b1']).sigmoid()
        a2 = (a1.dot(self.params['w2']) + self.params['b2']).sigmoid()
        a3 = (a2.dot(self.params['w3']) + self.params['b3']).sigmoid()
        x = a3.dot(self.params['w4']) + self.params['b4']
        return x
        
def sample_data():
    """
    Data generation process. Unknown to the model.
    """
    X, _ = datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.1)
    data_sample = np.stack([X[:, 0], X[:, 2]], axis=1)
    data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
    data_sample = data_sample * 2. - 1.
    data_sample = Tensor(data_sample)
    return data_sample

def sample_model(dec):
    z = nabla.randn((BATCH_SIZE, LATENT_DIM))
    model_sample = dec(z)
    return model_sample

def encode_decode(enc, dec, data_sample):
    z_mean, z_logvar = enc(data_sample)
    z = z_mean + E**(z_logvar * 0.5) * nabla.randn((BATCH_SIZE, LATENT_DIM))  # Reparam trick
    recon = dec(z)
    return z_mean, z_logvar, recon

def mse_loss(pred, gt):
    loss = ((pred - gt)**2).mean()
    return loss

def kl_loss(z_mean, z_logvar):
    loss = (-z_mean**2 - E**z_logvar + 1).mean() * (-0.5)
    return loss


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
    nabla.enable_grad(False)
    model_sample = sample_model(dec)
    nabla.enable_grad(True)
    fig, ax = plt.subplots()
    data_sample_plot = ax.plot(data_sample.data[:,0], data_sample.data[:,1], c='tab:blue', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample.data[:,0], model_sample.data[:,1], c='tab:red', marker='.', ls='', label='Model')[0]
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
        enc = zero_grad(enc)
        dec = zero_grad(dec)

        # Sample and viz
        losses_input_recon.append(loss_input_recon.data.squeeze()); losses_latent_prior.append(loss_latent_prior.data.squeeze())
        if it % 10 == 0:
            data_sample_plot.set_xdata(data_sample.data[:,0]); data_sample_plot.set_ydata(data_sample.data[:,1])
            nabla.enable_grad(False)
            model_sample = sample_model(dec)
            nabla.enable_grad(True)
            model_sample_plot.set_xdata(model_sample.data[:,0]); model_sample_plot.set_ydata(model_sample.data[:,1])
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
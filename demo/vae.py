""" 
Demonstration of the universal approximation capability of an NN.
Built using `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm

sys.path.append("../nabla_python")
from nabla import Tensor


# ---
# Constants
E = 2.718


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 128
BATCH_SIZE = 512
ITERS = 1000
BETA = 1e1


# ---
# Utils

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

def sample_model(model):
    latents = np.random.randn(LATENT_DIM, BATCH_SIZE)
    latents = Tensor(latents)
    model_sample = model.decode(latents)
    return model_sample

def encode_decode(model, data_sample):
    z_mean, z_logvar = model.encode(data_sample)
    z = z_mean + Tensor(np.array(E)) ** (z_logvar * Tensor(np.array(0.5))) * Tensor(np.random.randn(LATENT_DIM, BATCH_SIZE))  # Reparam trick
    recon = model.decode(z)
    return z_mean, z_logvar, recon

class AutoEncoder:
    def __init__(self):
        self.encoder_params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, 2)), requires_grad=True),
        'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),
        'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),
        'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4_mean': Tensor(np.random.normal(size=(LATENT_DIM, HIDDEN_DIM)), requires_grad=True),
        'b4_mean': Tensor(np.random.normal(size=(LATENT_DIM, 1)), requires_grad=True),
        'w4_logvar': Tensor(np.random.normal(size=(LATENT_DIM, HIDDEN_DIM)), requires_grad=True),
        'b4_logvar': Tensor(np.random.normal(size=(LATENT_DIM, 1)), requires_grad=True)
        }
        self.decoder_params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, LATENT_DIM)), requires_grad=True),
        'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),
        'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True),
        'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4': Tensor(np.random.normal(size=(2, HIDDEN_DIM)), requires_grad=True),
        'b4': Tensor(np.random.normal(size=(2, 1)), requires_grad=True)
        }
    def encode(self, x):
        a1 = (self.encoder_params['w1'].dot(x) + self.encoder_params['b1']).sigmoid()
        a2 = (self.encoder_params['w2'].dot(a1) + self.encoder_params['b2']).sigmoid()
        a3 = (self.encoder_params['w3'].dot(a2) + self.encoder_params['b3']).sigmoid()
        z_mean = self.encoder_params['w4_mean'].dot(a3) + self.encoder_params['b4_mean']
        z_logvar = self.encoder_params['w4_logvar'].dot(a3) + self.encoder_params['b4_logvar']
        return z_mean, z_logvar
    def decode(self, z):
        a1 = (self.decoder_params['w1'].dot(z) + self.decoder_params['b1']).sigmoid()
        a2 = (self.decoder_params['w2'].dot(a1) + self.decoder_params['b2']).sigmoid()
        a3 = (self.decoder_params['w3'].dot(a1) + self.decoder_params['b3']).sigmoid()
        x = self.decoder_params['w4'].dot(a2) + self.decoder_params['b4']
        return x

def mse_loss(pred, gt):
    loss = (pred - gt) ** Tensor(np.array(2.))
    loss = loss.sum() / Tensor(np.array(gt.shape[1]))
    return loss

def kl_loss(z_mean, z_logvar):
    loss = Tensor(np.array(-0.5)) * (Tensor(np.array(1)) - z_mean ** Tensor(np.array(2)) - Tensor(np.array(E)) ** z_logvar).sum()
    loss = loss / Tensor(np.array(z_mean.shape[1]))
    return loss

def update_params_and_zero_grad(model, lr):
    for param in model.encoder_params.values():
        param.data = param.data - lr*param.grad
        param.grad = np.zeros_like(param.grad)
    for param in model.decoder_params.values():
        param.data = param.data - lr*param.grad
        param.grad = np.zeros_like(param.grad)
    return model

# ---
# Main function
def main():

    # Init model
    model = AutoEncoder()

    # Init viz
    losses_input_recon, losses_latent_prior = [], []
    data_sample = sample_data()
    model_sample = sample_model(model)
    fig, ax = plt.subplots()
    data_sample_plot = ax.plot(data_sample.data[0, :], data_sample.data[1, :], c='tab:blue', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample.data[0, :], model_sample.data[1, :], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(ITERS)):

        # Update model
        data_sample = sample_data()
        latents_mean, latents_logvar, recon = encode_decode(model, data_sample)
        loss_input_recon = mse_loss(recon, data_sample)
        loss_latent_prior = kl_loss(latents_mean, latents_logvar)
        loss = loss_input_recon + Tensor(np.array(BETA)) * loss_latent_prior
        loss.backward()
        model = update_params_and_zero_grad(model, lr=0.001)

        # Sample and viz
        losses_input_recon.append(loss_input_recon.data.squeeze()); losses_latent_prior.append(loss_latent_prior.data.squeeze())
        if it % 10 == 0:
            data_sample_plot.set_xdata(data_sample.data[0, :]); data_sample_plot.set_ydata(data_sample.data[1, :])
            model_sample = sample_model(model)
            model_sample_plot.set_xdata(model_sample.data[0, :]); model_sample_plot.set_ydata(model_sample.data[1, :])
            fig.canvas.draw(); fig.canvas.flush_events()

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses_input_recon, label='Recon'); ax.plot(losses_latent_prior, label='Prior')
    ax.set_title("Training loss"); ax.legend(); plt.show()


# ---
# Run
if __name__ == '__main__':
    main()
"""
Demo of a Denoising Diffusion Probabilistic Model (Ho et al. NeurIPS 2020) on 2D swiss-roll toy dataset.
"""

import sys
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../python")
import nabla
from nabla import Tensor
from utils import AdamOptimizer, zero_grad



# ---
# Config
HIDDEN_DIM = 256
NUM_DIFFUSION_STEPS = 1000
BETA_T = 1e-4
BATCH_SIZE = 128
NUM_ITERS = 5000


# ---
# Precompute coeff schedules
beta_schedule = np.linspace(1e-6, BETA_T, NUM_DIFFUSION_STEPS) # Linear schedule
sigma_schedule = np.sqrt(beta_schedule)
alpha_schedule = 1 - beta_schedule
alpha_bar_schedule = np.empty_like(alpha_schedule)
alpha_bar_schedule[0] = alpha_schedule[0]
for t in range(1, NUM_DIFFUSION_STEPS):
    alpha_bar_schedule[t] = np.prod(alpha_schedule[:t])


# ---
# Utils

class NoiseModel:
    def __init__(self):
        self.params = {
        'w1': nabla.randn((3, HIDDEN_DIM), requires_grad=True), 'b1': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w2': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True), 'b2': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w3': nabla.randn((HIDDEN_DIM, HIDDEN_DIM), requires_grad=True), 'b3': nabla.zeros((1, HIDDEN_DIM), requires_grad=True),
        'w4': nabla.randn((HIDDEN_DIM, 2), requires_grad=True),          'b4': nabla.zeros((1, 2), requires_grad=True),
        }
    def __call__(self, x, t):
        t = t / NUM_DIFFUSION_STEPS
        t = t * 2. - 1.
        input = Tensor(np.concatenate((x.data, t.data), axis=1))
        a1 = (input.dot(self.params['w1']) + self.params['b1']).sigmoid()
        a2 = (a1.dot(self.params['w2']) + self.params['b2']).sigmoid()
        a3 = (a2.dot(self.params['w3']) + self.params['b3']).sigmoid()
        output = a3.dot(self.params['w4']) + self.params['b4']
        return output

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

def sample_model(noise_model):
     
    model_sample = nabla.randn((BATCH_SIZE, 2))

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):

        z = nabla.randn(model_sample.shape) if t > 0 else nabla.zeros(model_sample.shape)
        t_batch = np.full((BATCH_SIZE, 1), t)
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]        
        t_batch = Tensor(t_batch)
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (model_sample - noise_pred * (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * (1 / np.sqrt(alpha_t)) + \
                        z * sigma_t

    return model_sample

def show_reverse_diffusion(noise_model):
    
    num_samples = BATCH_SIZE * 8
    model_sample = nabla.randn((num_samples, 2))

    fig, ax = plt.subplots()
    model_sample_plot = ax.plot(model_sample.data[:,0], model_sample.data[:,1], c='tab:red', marker='.', ls='', alpha=0.5)[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Diffusion Model (reverse diffusion process)")
    fig.tight_layout(); plt.ion(); plt.show()    

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):
        
        z = nabla.randn(model_sample.shape) if t > 0 else nabla.zeros(model_sample.shape)
        t_batch = np.full((1, num_samples), t)
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]
        t_batch = Tensor(t_batch)
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (model_sample - noise_pred * (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * (1 / np.sqrt(alpha_t)) + \
                        z * sigma_t

        model_sample_plot.set_xdata(model_sample.data[:,0])
        model_sample_plot.set_ydata(model_sample.data[:,1])
        fig.canvas.draw(); fig.canvas.flush_events()
        # fig.savefig(f"./outputs/diffusion/{str(NUM_DIFFUSION_STEPS - 1 - t).zfill(5)}.png")
    
    plt.ioff()

def criterion(data_sample, t_batch, noise_model):
    alpha_bar_t = alpha_bar_schedule[t_batch.data]
    std_noise = nabla.randn(data_sample.shape)
    noise_pred = noise_model(data_sample * np.sqrt(alpha_bar_t) + std_noise * np.sqrt(1 - alpha_bar_t), t_batch)
    loss = ((std_noise - noise_pred) ** 2).sum()
    return loss

# ---
# Main function
def main():

    # Model and optimizer
    noise_model = NoiseModel()
    opt = AdamOptimizer(noise_model, lr=1e-4)

    # Visualization objects
    losses = []
    data_sample = sample_data()
    nabla.enable_grad(False)
    model_sample = sample_model(noise_model)
    nabla.enable_grad(True)
    fig, ax = plt.subplots()
    data_sample_plot = ax.scatter(data_sample.data[:,0], data_sample.data[:,1], c='tab:blue', marker='.', label='Data')
    model_sample_plot = ax.plot(model_sample.data[:,0], model_sample.data[:,1], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(NUM_ITERS)):
    
        # Update noise model
        data_sample = sample_data()
        t_batch = nabla.randint(0, NUM_DIFFUSION_STEPS, (BATCH_SIZE, 1))
        loss = criterion(data_sample, t_batch, noise_model)
        loss.backward()
        noise_model = opt.step(noise_model)
        noise_model = zero_grad(noise_model)
        
        # Sample and viz
        losses.append(loss.data.squeeze())
        if it % 1000 == 0:
            nabla.enable_grad(False)
            model_sample = sample_model(noise_model)
            nabla.enable_grad(True)
            model_sample_plot.set_xdata(model_sample.data[:,0])
            model_sample_plot.set_ydata(model_sample.data[:,1])
            fig.canvas.draw(); fig.canvas.flush_events() 

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses); ax.set_title("Training loss"); plt.show()
    show_reverse_diffusion(noise_model)


# ---
# Run
if __name__ == '__main__':
    main()
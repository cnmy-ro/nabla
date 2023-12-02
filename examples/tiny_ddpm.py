"""
Demo of a Denoising Diffusion Probabilistic Model (Ho et al. NeurIPS 2020) on 2D swiss-roll toy dataset.
"""

import sys
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../nabla_python")
import nabla
from nabla import Tensor
from utils import AdamOptimizer



# ---
# Config
HIDDEN_DIM = 256
NUM_DIFFUSION_STEPS = 1000
BETA_T = 1e-4
BATCH_SIZE = 64
NUM_ITERS = 10000


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
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, 3)), requires_grad=True), 'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4': Tensor(np.random.normal(size=(2, HIDDEN_DIM)), requires_grad=True),          'b4': Tensor(np.random.normal(size=(2, 1)), requires_grad=True),
        }
    def __call__(self, x, t):
        t = t / NUM_DIFFUSION_STEPS
        t = t * 2. - 1.
        input = Tensor(np.concatenate((x.data, t.data), axis=0))
        a1 = (self.params['w1'].dot(input) + self.params['b1']).sigmoid()
        a2 = (self.params['w2'].dot(a1) + self.params['b2']).sigmoid()
        a3 = (self.params['w3'].dot(a2) + self.params['b3']).sigmoid()
        output = self.params['w4'].dot(a3) + self.params['b4']
        return output

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

def sample_model(noise_model):
     
    model_sample = Tensor(np.random.randn(2, BATCH_SIZE))

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):

        z = Tensor(np.random.randn(*model_sample.shape)) if t > 0 else Tensor(np.zeros(model_sample.shape))
        t_batch = np.full((1, BATCH_SIZE), t)
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]
        t_batch = Tensor(t_batch)
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (model_sample - noise_pred * (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * (1 / np.sqrt(alpha_t)) + \
                        z * sigma_t

    return model_sample

def show_reverse_diffusion(noise_model):
    
    model_sample = Tensor(np.random.randn(2, BATCH_SIZE))

    fig, ax = plt.subplots()
    model_sample_plot = ax.plot(model_sample.data[0, :], model_sample.data[1, :], c='tab:red', marker='.', ls='')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Reverse diffusion process")        
    fig.tight_layout(); plt.ion(); plt.show()    

    for t in range(NUM_DIFFUSION_STEPS - 1, 0, -1):
        
        z = Tensor(np.random.randn(*model_sample.shape)) if t > 0 else Tensor(np.zeros(model_sample.shape))
        t_batch = np.full((1, BATCH_SIZE), t)
        sigma_t, alpha_t, alpha_bar_t = sigma_schedule[t_batch], alpha_schedule[t_batch], alpha_bar_schedule[t_batch]
        t_batch = Tensor(t_batch)
        noise_pred = noise_model(model_sample, t_batch)
        model_sample = (model_sample - noise_pred * (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * (1 / np.sqrt(alpha_t)) + \
                        z * sigma_t

        model_sample_plot.set_xdata(model_sample.data[0, :])
        model_sample_plot.set_ydata(model_sample.data[1, :])
        fig.canvas.draw(); fig.canvas.flush_events()
    
    plt.ioff()

def criterion(data_sample, t_batch, noise_model):
    alpha_bar_t = alpha_bar_schedule[t_batch.data]
    std_noise = Tensor(np.random.randn(*data_sample.shape))
    noise_pred = noise_model(data_sample * np.sqrt(alpha_bar_t) + std_noise * np.sqrt(1 - alpha_bar_t), t_batch)
    loss = ((std_noise - noise_pred) ** 2).sum()
    return loss

def zero_grad(model):
    for param in model.params.values():
        param.grad = np.zeros_like(param.grad)
        param.parents = None
    return model

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
    data_sample_plot = ax.scatter(data_sample.data[0, :], data_sample.data[1, :], c='tab:blue', marker='.', label='Data')
    model_sample_plot = ax.plot(model_sample.data[0, :], model_sample.data[1, :], c='tab:red', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(NUM_ITERS)):
    
        # Update noise model
        data_sample = sample_data()
        t_batch = Tensor(np.random.randint(0, NUM_DIFFUSION_STEPS, (1, BATCH_SIZE)))
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
            model_sample_plot.set_xdata(model_sample.data[0, :])
            model_sample_plot.set_ydata(model_sample.data[1, :])
            fig.canvas.draw(); fig.canvas.flush_events() 

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses); ax.set_title("Training loss"); plt.show()
    show_reverse_diffusion(noise_model)


# ---
# Run
if __name__ == '__main__':
    main()
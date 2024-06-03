""" 
Generative adversarial networks built with `nabla`.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm

sys.path.append("../pynabla")
import nabla
from nabla import Tensor
from utils import AdamOptimizer


# ---
# Constants
E = 2.718


# ---
# Config
LATENT_DIM = 2
HIDDEN_DIM = 256
BATCH_SIZE = 64
DIS_ITERS = 10
ITERS = 3000
LR_G, LR_D = 1e-4, 1e-4


# ---
# Utils

class Generator:
    def __init__(self):
        self.params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, LATENT_DIM)), requires_grad=True), 'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4': Tensor(np.random.normal(size=(2, HIDDEN_DIM)), requires_grad=True),          'b4': Tensor(np.random.normal(size=(2, 1)), requires_grad=True),
        }
    def __call__(self, z):
        a1 = (self.params['w1'].dot(z) + self.params['b1']).sigmoid()
        a2 = (self.params['w2'].dot(a1) + self.params['b2']).sigmoid()
        a3 = (self.params['w3'].dot(a2) + self.params['b3']).sigmoid()
        y = self.params['w4'].dot(a3) + self.params['b4']
        return y

class Discriminator:
    def __init__(self):
        self.params = {
        'w1': Tensor(np.random.normal(size=(HIDDEN_DIM, 2)), requires_grad=True),          'b1': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w2': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b2': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w3': Tensor(np.random.normal(size=(HIDDEN_DIM, HIDDEN_DIM)), requires_grad=True), 'b3': Tensor(np.random.normal(size=(HIDDEN_DIM, 1)), requires_grad=True),
        'w4': Tensor(np.random.normal(size=(1, HIDDEN_DIM)), requires_grad=True),          'b4': Tensor(np.random.normal(size=(1, 1)), requires_grad=True),
        }
    def __call__(self, x):
        a1 = (self.params['w1'].dot(x) + self.params['b1']).sigmoid()
        a2 = (self.params['w2'].dot(a1) + self.params['b2']).sigmoid()
        a3 = (self.params['w3'].dot(a2) + self.params['b3']).sigmoid()
        y = (self.params['w4'].dot(a3) + self.params['b4']).sigmoid()
        return y

def sample_data(num_samples=BATCH_SIZE):
    """
    Data generation process. Unknown to the model.
    """
    X, _ = datasets.make_swiss_roll(n_samples=num_samples, noise=0.1)
    data_sample = np.stack([X[:, 0], X[:, 2]], axis=1)
    data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
    data_sample = data_sample * 2. - 1.
    data_sample = data_sample.T  # To shape (len, batch)
    data_sample = Tensor(data_sample)
    return data_sample

def sample_model(gen, num_samples=BATCH_SIZE):    
    z = Tensor(np.random.randn(LATENT_DIM, num_samples))
    model_sample = gen(z)
    return model_sample

def add_noise(x, iter_counter):
    noise_level = 0.01
    return x + Tensor(np.random.randn(*x.shape)) * (1. - iter_counter/ITERS) * noise_level

def nsgan_loss(pred, is_real):
    if is_real: loss = -( pred.log().mean() )
    else:       loss = -( (Tensor(np.array(1.)) - pred).log().mean() )
    return loss

def zero_grad(model):
    for param in model.params.values():
        param.grad = np.zeros_like(param.grad)
        param.parents = None
    return model

def compute_disriminator_landscape(dis):
    nabla.enable_grad(False)
    xx, yy = np.meshgrid(np.arange(-1.2, 1.2, 0.01), np.arange(-1.2, 1.2, 0.01))
    xx_t = xx.flatten().T
    yy_t = yy.flatten().T
    pred_landscape = dis(Tensor(np.stack((xx_t, yy_t), axis=0)))
    pred_landscape = pred_landscape.data.reshape(xx.shape)
    nabla.enable_grad(True)
    return xx, yy, pred_landscape


# ---
# Main function

def main():

    # Init model and opts
    gen = Generator()
    dis = Discriminator()
    opt_g = AdamOptimizer(gen, lr=LR_G)
    opt_d = AdamOptimizer(dis, lr=LR_D)

    # Init viz
    losses_g, losses_d = [], []
    data_sample = sample_data()
    nabla.enable_grad(False)
    model_sample = sample_model(gen)
    nabla.enable_grad(True)
    xx, yy, pred_landscape = compute_disriminator_landscape(dis)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, pred_landscape)
    data_sample_plot = ax.plot(data_sample.data[0, :], data_sample.data[1, :], c='purple', marker='.', ls='', label='Data')[0]
    model_sample_plot = ax.plot(model_sample.data[0, :], model_sample.data[1, :], c='orangered', marker='.', ls='', label='Model')[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_title("Samples")
    fig.legend(); fig.tight_layout(); plt.ion(); plt.show()

    # Training loop
    for it in tqdm(range(ITERS)):

        # Update D
        for _ in range(DIS_ITERS):
            data_sample = add_noise(sample_data(), it)
            model_sample = add_noise(sample_model(gen), it)
            model_sample.parents = None  # Detach
            loss_d = nsgan_loss(dis(data_sample), is_real=True) + nsgan_loss(dis(model_sample), is_real=False)
            loss_d.backward()
            dis = opt_d.step(dis)
            dis = zero_grad(dis)

        # Update G
        model_sample = sample_model(gen)
        loss_g = nsgan_loss(dis(model_sample), is_real=True)
        loss_g.backward()
        gen = opt_g.step(gen)
        gen = zero_grad(gen)
        dis = zero_grad(dis)

        # Sample and viz
        losses_g.append(loss_g.data.squeeze()); losses_d.append(loss_d.data.squeeze())
        if it % 10 == 0:
            num_samples = BATCH_SIZE * 4
            data_sample = sample_data(num_samples)
            model_sample = sample_model(gen, num_samples)
            xx, yy, pred_landscape = compute_disriminator_landscape(dis)
            ax.contourf(xx, yy, pred_landscape, cmap='cividis')
            data_sample_plot.set_xdata(data_sample.data[0, :]); data_sample_plot.set_ydata(data_sample.data[1, :])
            model_sample_plot.set_xdata(model_sample.data[0, :]); model_sample_plot.set_ydata(model_sample.data[1, :])
            fig.canvas.draw(); fig.canvas.flush_events()
            fig.savefig(f"./outputs/gan/{str(it).zfill(5)}.png")
    
    # Plot loss curves
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(losses_g, label='G'); ax.plot(losses_d, label='D')
    ax.set_title("Training loss"); ax.legend(); plt.show()


# ---
# Run
if __name__ == '__main__':
    main()
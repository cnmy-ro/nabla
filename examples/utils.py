import numpy as np


class AdamOptimizer:
    def __init__(self, model, lr, beta_1=0.9, beta_2=0.99):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_dict = {k: np.zeros_like(param) for k, param in model.params.items()}
        self.v_dict = {k: np.zeros_like(param) for k, param in model.params.items()}
        self.t = 1
    def step(self, model):
        for k, param in model.params.items():
            self.m_dict[k] = self.beta_1 * self.m_dict[k] + (1 - self.beta_1) * param.grad
            self.v_dict[k] = self.beta_2 * self.v_dict[k] + (1 - self.beta_2) * param.grad**2
            m_hat = self.m_dict[k] / (1 - self.beta_1**self.t)
            v_hat = self.v_dict[k] / (1 - self.beta_2**self.t)
            param.data = param.data - self.lr * m_hat / np.sqrt(v_hat + 1e-8)
            param.grad = np.zeros_like(param.data)
            self.t += 1
        return model
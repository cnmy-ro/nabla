from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np


# ---
# Core classes

class Tensor:
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.grad = np.zeros_like(data)  # d_root_tensor / d_self_tensor
        self.shape = data.shape
        self.requires_grad = requires_grad
        self.prev = None
    
    def backward(self, grad: np.ndarray = np.array([1.])):
        self.grad += grad        
        if self.prev is not None:  # Depth-first tree traversal
            fn, fn_vars = self.prev[0], self.prev[1]
            fn_vars_vjp = fn.vjp(self.grad, self, *fn_vars)
            for i in range(len(fn_vars)):
                if fn_vars[i].requires_grad:
                    fn_var_grad = fn_vars_vjp[i]
                    fn_vars[i].backward(fn_var_grad)

    def __neg__(self): return Neg()(self)
    def __add__(self, other): return Add()(self, other)
    def __sub__(self, other): return Sub()(self, other)
    def __mul__(self, other): return Mul()(self, other)
    def __truediv__(self, other): return Div()(self, other)
    def __pow__(self, other): return Pow()(self, other)
    def dot(self, other): return Dot()(self, other)
    def sum(self): return Sum()(self)
    def relu(self): return ReLU()(self)
    def sigmoid(self): return Sigmoid()(self)
    def tanh(self): return Tanh()(self)

class Operator(ABC):
    
    def __call__(self, *args: Tensor) -> Tensor:
        y = self.fx(*args)
        y.prev = [self, args]
        return y

    @abstractmethod
    def fx(self, *args: Tensor) -> Tensor:
        """ Forward function """
        ...

    @abstractmethod
    def vjp(self, grad: np.ndarray, result: Tensor, *args: Tensor) -> List[np.ndarray]:
        """
        Vector-Jacobian product.
        Implicitly computes J-dot-grad without having to materialize the massive J.
        """
        ...


# ---
# Unary ops

class Neg(Operator):
    def fx(self, x): return Tensor(-x.data, requires_grad=True)
    def vjp(self, grad, result, x): return [-grad]

class Sum(Operator):
    def fx(self, x): return Tensor(x.data.sum(), requires_grad=True)
    def vjp(self, grad, result, x): return [grad]

class ReLU(Operator):
    def fx(self, x): return Tensor(np.maximum(x.data, np.zeros_like(x.data)), requires_grad=True)
    def vjp(self, grad, result, x): return [grad * (result.data > 0.).astype(float)]

class Sigmoid(Operator):
    def _sigma(self, x): return 1. / (1. + np.exp(-x))
    def fx(self, x): return Tensor(self._sigma(x.data), requires_grad=True)
    def vjp(self, grad, result, x): return [grad * self._sigma(x.data) * (1. - self._sigma(x.data))]

class Tanh(Operator):
    def fx(self, x): return Tensor(np.tanh(x.data), requires_grad=True)
    def vjp(self, grad, result, x): return [grad * (1. - np.tanh(x.data)**2)]


# ---
# Binary ops

class Add(Operator):
    def fx(self, x1, x2): return Tensor(x1.data + x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad, grad]

class Sub(Operator):
    def fx(self, x1, x2): return Tensor(x1.data - x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad, -grad]

class Mul(Operator):
    def fx(self, x1, x2): return Tensor(x1.data * x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad * x2.data, grad * x1.data]

class Div(Operator):
    def fx(self, x1, x2): return Tensor(x1.data / x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad * (1. / x2.data), grad * x1.data * (-1. / (x2.data**2))]

class Pow(Operator):
    def fx(self, x1, x2): return Tensor(x1.data ** x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad * x2.data * x1.data ** (x2.data - 1.)]

class Dot(Operator):
    def fx(self, x1, x2): return Tensor(x1.data @ x2.data, requires_grad=True)
    def vjp(self, grad, result, x1, x2): return [grad @ x2.data.T, x1.data.T @ grad]
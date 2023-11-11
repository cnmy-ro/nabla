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
            op, op_args = self.prev[0], self.prev[1]
            op_args_vjp = op.vjp(self, *op_args)
            for i in range(len(op_args)):
                if op_args[i].requires_grad:
                    op_arg_grad = op_args_vjp[i]
                    op_args[i].backward(op_arg_grad)

    def __neg__(self):            return Neg()(self)
    def __add__(self, other):     return Add()(self, other)
    def __sub__(self, other):     return Sub()(self, other)
    def __mul__(self, other):     return Mul()(self, other)
    def __truediv__(self, other): return Div()(self, other)
    def __pow__(self, other):     return Pow()(self, other)
    def dot(self, other):         return Dot()(self, other)
    def sum(self):                return Sum()(self)
    def relu(self):               return ReLU()(self)
    def sigmoid(self):            return Sigmoid()(self)
    def tanh(self):               return Tanh()(self)

class Operator(ABC):
    
    def __call__(self, *args: Tensor) -> Tensor:
        y = self.fx(*args)
        y = Tensor(y, requires_grad=True)
        y.prev = [self, args]
        return y

    @abstractmethod
    def fx(self, *args: Tensor) -> np.ndarray:
        """ Forward evaluation function """
        ...

    @abstractmethod
    def vjp(self, eval: Tensor, *args: Tensor) -> List[np.ndarray]:
        """
        Vector-Jacobian product.
        Implicitly computes J-dot-grad without having to materialize the massive J.
        
        Args:
            eval: result tensor of this op (i.e. of the fx function)
            *args: argument tensors to this op
        Returns:
            grads of this op's argument tensors
        """
        ...

# ---
# Unary ops

class Neg(Operator):
    def fx(self, x):        return -x.data
    def vjp(self, eval, x): return [-eval.grad]

class Sum(Operator):
    def fx(self, x):        return x.data.sum()
    def vjp(self, eval, x): return [eval.grad]

class ReLU(Operator):
    def fx(self, x):        return np.maximum(x.data, np.zeros_like(x.data))
    def vjp(self, eval, x): return [eval.grad * (eval.data > 0.).astype(float)]

class Sigmoid(Operator):
    def _sigma(self, x):    return 1. / (1. + np.exp(-x))
    def fx(self, x):        return self._sigma(x.data)
    def vjp(self, eval, x): return [eval.grad * self._sigma(x.data) * (1. - self._sigma(x.data))]

class Tanh(Operator):
    def fx(self, x):        return np.tanh(x.data)
    def vjp(self, eval, x): return [eval.grad * (1. - np.tanh(x.data)**2)]

# ---
# Binary ops

class Add(Operator):
    def fx(self, x1, x2):        return x1.data + x2.data
    def vjp(self, eval, x1, x2): return [eval.grad, eval.grad]

class Sub(Operator):
    def fx(self, x1, x2):        return x1.data - x2.data
    def vjp(self, eval, x1, x2): return [eval.grad, -eval.grad]

class Mul(Operator):
    def fx(self, x1, x2):        return x1.data * x2.data
    def vjp(self, eval, x1, x2): return [eval.grad * x2.data, eval.grad * x1.data]

class Div(Operator):
    def fx(self, x1, x2):        return x1.data / x2.data
    def vjp(self, eval, x1, x2): return [eval.grad * (1. / x2.data), eval.grad * x1.data * (-1. / (x2.data**2))]

class Pow(Operator):
    def fx(self, x1, x2):        return x1.data ** x2.data
    def vjp(self, eval, x1, x2): return [eval.grad * x2.data * x1.data ** (x2.data - 1.)]

class Dot(Operator):
    def fx(self, x1, x2):        return x1.data @ x2.data
    def vjp(self, eval, x1, x2): return [eval.grad @ x2.data.T, x1.data.T @ eval.grad]
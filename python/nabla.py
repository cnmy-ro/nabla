from abc import ABC, abstractmethod
from typing import List
import numpy as np

_grad_enabled = True  # Global switch


# ---
# Core functions

def enable_grad(flag):
    global _grad_enabled
    _grad_enabled = flag

def show_dag(tensor):
    # TODO: trace the history of the given tensor and show as DAG
    pass

# ---
# Core classes

class Tensor:
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data)  # d_root / d_self
        self.op = None  # Operator that produced this tensor in the computational graph.
        self.parents = None  # List [op_args]. Records the parent node(s) in the computational graph.
        self.shape = data.shape
    
    def backward(self, grad: np.ndarray = np.array([[1.]])): # Backprop function
        if not _grad_enabled: raise RuntimeError("Called backward() but gradient computation is disabled.")
        self._accumulate_grad(grad)
        if self.op is not None:  # Recursive depth-first tree traversal
            parent_vjps = self.op.vjp(self, *self.parents)
            for i in range(len(self.parents)):
                if self.parents[i].requires_grad:
                    parent_grad = parent_vjps[i]
                    self.parents[i].backward(parent_grad)

    def detach(self):
        self.op = self.parents = None

    def _accumulate_grad(self, grad):
        bc_dims = []  # If this tensor had size=1 in certain dims and was broadcast during forward pass, its grad will have size>1 in those dims
        for dim in range(len(self.shape)):
            if self.shape[dim] == 1 and grad.shape[dim] > 1:
                bc_dims.append(dim)
        if len(bc_dims) > 0: grad = np.apply_over_axes(np.sum, grad, bc_dims)
        self.grad += grad

    def __str__(self):            return self.data.__str__()
    def __repr__(self):           return self.data.__repr__()
    def __neg__(self):            return Neg()(self)
    def __add__(self, other):     return Add()(self, other)
    def __sub__(self, other):     return Sub()(self, other)
    def __mul__(self, other):     return Mul()(self, other)
    def __truediv__(self, other): return Div()(self, other)
    def __pow__(self, other):     return Pow()(self, other)
    def dot(self, other):         return Dot()(self, other)
    def sum(self):                return Sum()(self)
    def mean(self):               return Sum()(self) / Tensor(np.prod(np.array(self.shape)))
    def log(self):                return Log()(self)
    def relu(self):               return ReLU()(self)
    def leaky_relu(self, alpha):  return LeakyReLU(alpha)(self)
    def sigmoid(self):            return Sigmoid()(self)
    def tanh(self):               return Tanh()(self)    

class Operator(ABC):
    
    def __call__(self, *args: Tensor) -> Tensor:
        
        # Auto typecast float/int into Tensor. Limitation: During usage, the 1st argument must be a Tensor (e.g. "Tensor + float")
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], float) or isinstance(args[i], int):
                args[i] = Tensor(np.array(args[i]))

        y = self.fx(*args)
        y = Tensor(y)
        if _grad_enabled:
            y.requires_grad = True
            y.op = self
            y.parents = args
        return y

    @abstractmethod
    def fx(self, *args: Tensor) -> np.ndarray:
        """ Forward function """
        ...

    @abstractmethod
    def vjp(self, y: Tensor, *args: Tensor) -> List[np.ndarray]:
        """
        Vector-Jacobian product.
        Implicitly computes J.T-dot-grad without having to materialize the massive J.
        
        Args:
            y: result tensor of this op (i.e. of the fx function)
            *args: argument tensors to this op
        Returns:
            grads of this op's argument tensors
        """
        ...

# ---
# Point-wise unary ops

class Neg(Operator):
    def fx(self, x):     return -x.data
    def vjp(self, y, x):
        x_grad = y.grad * (-1.)
        return [x_grad]

class Log(Operator):
    def fx(self, x):     return np.log(x.data)
    def vjp(self, y, x):
        x_grad = y.grad * (1. / (x.data + 1e-8))
        return [x_grad]

class ReLU(Operator):
    def fx(self, x):     return np.maximum(x.data, np.zeros_like(x.data))
    def vjp(self, y, x): 
        x_grad = y.grad * (x.data > 0.).astype(float)
        return [x_grad]

class LeakyReLU(Operator):
    def __init__(self, alpha=0.2): self.alpha = alpha
    def fx(self, x):
        y = x.data.copy()
        y[x.data < 0] *= self.alpha
        return y
    def vjp(self, y, x):
        x_grad = y.grad
        x_grad[x.data < 0] *= self.alpha
        return [x_grad]

class Sigmoid(Operator):
    def _sigma(self, x): return 1. / (1. + np.exp(-x))
    def fx(self, x):     return self._sigma(x.data)
    def vjp(self, y, x):
        x_grad = y.grad * self._sigma(x.data) * (1. - self._sigma(x.data))
        return [x_grad]

class Tanh(Operator):
    def fx(self, x):     return np.tanh(x.data)
    def vjp(self, y, x): 
        x_grad = y.grad * (1. - np.tanh(x.data)**2)
        return [x_grad]

# ---
# Point-wise binary ops

class Add(Operator):
    def fx(self, x1, x2): return x1.data + x2.data
    def vjp(self, y, x1, x2):
        x1_grad, x2_grad = y.grad, y.grad
        return [x1_grad, x2_grad]

class Sub(Operator):
    def fx(self, x1, x2): return x1.data - x2.data
    def vjp(self, y, x1, x2):
        x1_grad, x2_grad = y.grad, -y.grad
        return [x1_grad, x2_grad]

class Mul(Operator):
    def fx(self, x1, x2): return x1.data * x2.data
    def vjp(self, y, x1, x2):
        x1_grad, x2_grad = y.grad * x2.data, y.grad * x1.data
        return [x1_grad, x2_grad]

class Div(Operator):
    def fx(self, x1, x2): return x1.data / x2.data
    def vjp(self, y, x1, x2): 
        x1_grad, x2_grad = y.grad * (1. / x2.data), y.grad * x1.data * (-1. / (x2.data**2))
        return [x1_grad, x2_grad]

class Pow(Operator):
    def fx(self, x1, x2): return x1.data**x2.data
    def vjp(self, y, x1, x2):
        x1_grad = y.grad * x2.data * x1.data**(x2.data - 1.)
        if x2.requires_grad:
            x2_grad = y.grad * np.log(x1.data) * x1.data**x2.data
        else:
            x2_grad = np.zeros_like(x2.data)
        return [x1_grad, x2_grad]

# ---
# Shape-altering unary ops

class Sum(Operator):
    # TODO: sum along specified dim
    def fx(self, x):     return x.data.sum(keepdims=True)
    def vjp(self, y, x):
        x_grad = np.full(x.shape, y.grad * 1.)
        return [x_grad]

# ---
# Shape-altering binary ops

class Dot(Operator):
    def fx(self, x1, x2): return x1.data @ x2.data
    def vjp(self, y, x1, x2): 
        x1_grad, x2_grad = y.grad @ x2.data.T, x1.data.T @ y.grad
        return [x1_grad, x2_grad]

class Conv1D(Operator):
    def fx(self, x1, x2):
        # TODO
        pass
    def vjp(self, y, x1, x2): 
        # TODO
        pass

class Conv2D(Operator):
    def fx(self, x1, x2): 
        # TODO
        pass
    def vjp(self, y, x1, x2): 
        # TODO
        pass

# ---
# Shape ops
# TODO: reshape, flatten, permute, squeeze, unsqueeze

# ---
# Convenience functions

def zeros(shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)

def ones(shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad=requires_grad)

def rand(shape, requires_grad=False):
    return Tensor(np.random.rand(size=shape), requires_grad=requires_grad)

def randn(shape, requires_grad=False):
    return Tensor(np.random.normal(size=shape), requires_grad=requires_grad)

def randint(start, end, shape, requires_grad=False):
    return Tensor(np.random.randint(start, end, shape), requires_grad=requires_grad)
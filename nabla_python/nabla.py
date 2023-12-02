from abc import ABC, abstractmethod
import numpy as np

_enable_grad = True  # Global switch

# ---
# Core classes

class Tensor:
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data)  # d_root / d_self
        self.parents = None  # List [op, op_args]. Records the parent node(s) in the computational graph.
        self.shape = data.shape
    
    def backward(self, grad: np.ndarray = np.array([[1.]])): # Backprop function
        if not _enable_grad: raise RuntimeError("Called backward() but gradient computation is disabled.")
        self.grad += grad
        if self.parents is not None:  # Recursive depth-first tree traversal
            op, op_args = self.parents[0], self.parents[1]
            op_args_vjp = op.vjp(self, *op_args)
            for i in range(len(op_args)):
                if op_args[i].requires_grad:
                    op_arg_grad = op_args_vjp[i]
                    op_args[i].backward(op_arg_grad)

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
        y = Tensor(y, requires_grad=True)
        if _enable_grad:
            y.parents = [self, args]
        return y

    @abstractmethod
    def fx(self, *args: Tensor) -> np.ndarray:
        """ Forward function """
        ...

    @abstractmethod
    def vjp(self, y: Tensor, *args: Tensor) -> list[np.ndarray]:
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
# Unary ops

class Neg(Operator):
    def fx(self, x):     return -x.data
    def vjp(self, y, x):
        grad_x = y.grad * (-1.)
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class Sum(Operator):
    def fx(self, x):     return x.data.sum(keepdims=True)
    def vjp(self, y, x):
        grad_x = y.grad * 1.
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class Log(Operator):
    def fx(self, x):     return np.log(x.data)
    def vjp(self, y, x):
        grad_x = y.grad * (1. / (x.data + 1e-8))
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class ReLU(Operator):
    def fx(self, x):     return np.maximum(x.data, np.zeros_like(x.data))
    def vjp(self, y, x): 
        grad_x = y.grad * (x.data > 0.).astype(float)
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class LeakyReLU(Operator):
    def __init__(self, alpha):  self.alpha = alpha
    def fx(self, x):
        y = x.data.copy()
        y[x.data < 0.] *= self.alpha
        return y
    def vjp(self, y, x):
        grad_x = y.grad
        grad_x[x.data < 0.] *= self.alpha
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class Sigmoid(Operator):
    def _sigma(self, x): return 1. / (1. + np.exp(-x))
    def fx(self, x):     return self._sigma(x.data)
    def vjp(self, y, x):
        grad_x = y.grad * self._sigma(x.data) * (1. - self._sigma(x.data))
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

class Tanh(Operator):
    def fx(self, x):     return np.tanh(x.data)
    def vjp(self, y, x): 
        grad_x = y.grad * (1. - np.tanh(x.data)**2)
        grad_x = sum_grads_across_batch(x, grad_x)
        return [grad_x]

# ---
# Binary ops

class Add(Operator):
    def fx(self, x1, x2): return x1.data + x2.data
    def vjp(self, y, x1, x2):
        grad_x1, grad_x2 = y.grad, y.grad
        grad_x1, grad_x2 = sum_grads_across_batch(x1, grad_x1), sum_grads_across_batch(x2, grad_x2)
        return [grad_x1, grad_x2]

class Sub(Operator):
    def fx(self, x1, x2): return x1.data - x2.data
    def vjp(self, y, x1, x2):
        grad_x1, grad_x2 = y.grad, -y.grad
        grad_x1, grad_x2 = sum_grads_across_batch(x1, grad_x1), sum_grads_across_batch(x2, grad_x2)
        return [grad_x1, grad_x2]

class Mul(Operator):
    def fx(self, x1, x2): return x1.data * x2.data
    def vjp(self, y, x1, x2):
        grad_x1, grad_x2 = y.grad * x2.data, y.grad * x1.data
        grad_x1, grad_x2 = sum_grads_across_batch(x1, grad_x1), sum_grads_across_batch(x2, grad_x2)
        return [grad_x1, grad_x2]

class Div(Operator):
    def fx(self, x1, x2): return x1.data / x2.data
    def vjp(self, y, x1, x2): 
        grad_x1, grad_x2 = y.grad * (1. / x2.data), y.grad * x1.data * (-1. / (x2.data**2))
        grad_x1, grad_x2 = sum_grads_across_batch(x1, grad_x1), sum_grads_across_batch(x2, grad_x2)
        return [grad_x1, grad_x2]

class Pow(Operator):
    def fx(self, x1, x2): return x1.data**x2.data
    def vjp(self, y, x1, x2):
        grad_x1 = y.grad * x2.data * x1.data**(x2.data - 1.)
        grad_x1 = sum_grads_across_batch(x1, grad_x1)
        if x2.requires_grad:
            grad_x2 = y.grad * np.log(x1.data) * x1.data**x2.data
            grad_x2 = sum_grads_across_batch(x2, grad_x2)
        else:
            grad_x2 = np.zeros_like(x2.data)
        return [grad_x1, grad_x2]

class Dot(Operator):
    def fx(self, x1, x2): return x1.data @ x2.data
    def vjp(self, y, x1, x2): 
        grad_x1, grad_x2 = y.grad @ x2.data.T, x1.data.T @ y.grad
        grad_x1, grad_x2 = sum_grads_across_batch(x1, grad_x1), sum_grads_across_batch(x2, grad_x2)
        return [grad_x1, grad_x2]

# ---
# Utils

def sum_grads_across_batch(arg, grad_arg):
    if len(arg.shape) > 1:
        if arg.shape[1] == 1 and grad_arg.shape[1] > 1:
            grad_arg = grad_arg.sum(axis=1, keepdims=True)
    return grad_arg

def enable_grad(flag):
    global _enable_grad
    _enable_grad = flag
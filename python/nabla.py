from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import graphviz
import matplotlib.pyplot as plt

_grad_enabled = True   # Global switch
_tensor_namespace = [] # Global list of all tensor names


# ---
# Core functions

def enable_grad(flag):
    global _grad_enabled
    _grad_enabled = flag

def show_dag(tensor, view_img=True):
    graph = graphviz.Digraph(f"nabla_dag_{tensor.name}", format='png')
    def traverse_dag(tnsr):  # Build graph with recursive depth-first tree traversal
        if isinstance(tnsr.op, Stack) or isinstance(tnsr.op, Cat): parents = tnsr.parents[0]
        else:                                                      parents = tnsr.parents
        if parents is not None:
            for i in range(len(parents)):
                from_node = f"{(parents[i].op.__class__.__name__)} -> {parents[i].name}"
                to_node = f"{tnsr.op.__class__.__name__} -> {tnsr.name}"
                graph.edge(from_node, to_node, parents[i].name)
                traverse_dag(parents[i])
    traverse_dag(tensor)
    if view_img:
        graph.render(directory='/tmp', view=False)
        img = plt.imread(f"/tmp/nabla_dag_{tensor.name}.gv.png")  
        plt.imshow(img); plt.axis('off'); plt.show()
    else:
        print(graph.source)

# ---
# Core classes

class Tensor:
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.name = _generate_tensor_name()
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data)  # d_root / d_self
        self.op = None  # Operator that produced this tensor in the computational graph.
        self.parents = None  # List [op_args]. Records the parent node(s) in the computational graph.
        self.shape = data.shape
    
    def backward(self, grad: np.ndarray = np.array([1.])): # Backprop function
        if not _grad_enabled: raise RuntimeError("Called backward(), but gradient computation not enabled")
        self._accumulate_grad(grad)
        if self.op is not None:
            if isinstance(self.op, Stack) or isinstance(self.op, Cat): parents = self.parents[0]
            else:                                                      parents = self.parents
            parent_vjps = self.op.vjp(self, *parents)
            for i in range(len(parents)):
                if parents[i].requires_grad:
                    parent_grad = parent_vjps[i]
                    parents[i].backward(parent_grad)  # Recursive depth-first tree traversal    

    def _accumulate_grad(self, grad):
        bc_dims = []  # If this tensor had size=1 in certain dims and was broadcast during forward pass, its grad will have size>1 in those dims
        for dim in range(len(self.shape)):
            if self.shape[dim] == 1 and grad.shape[dim] > 1:
                bc_dims.append(dim)
        if len(bc_dims) > 0: grad = np.apply_over_axes(np.sum, grad, bc_dims)
        self.grad += grad

    def __str__(self):            return self.data.__str__()
    def __repr__(self):           return self.data.__repr__()
    def __getitem__(self, idx):   return Slice(idx)(self)
    def __neg__(self):            return Neg()(self)
    def __add__(self, other):     return Add()(self, other)
    def __sub__(self, other):     return Sub()(self, other)
    def __mul__(self, other):     return Mul()(self, other)
    def __truediv__(self, other): return Div()(self, other)
    def __pow__(self, other):     return Pow()(self, other)
    def dot(self, other):         return Dot()(self, other)
    def sum(self, dim=None):      return Sum(dim)(self)
    def mean(self, dim=None):     return Sum(dim)(self) / int(np.prod(np.array(self.shape))) if dim is None else Sum(dim)(self) / int(np.prod(np.array(self.shape)[dim]))
    def log(self):                return Log()(self)
    def relu(self):               return ReLU()(self)
    def leaky_relu(self, alpha):  return LeakyReLU(alpha)(self)
    def sigmoid(self):            return Sigmoid()(self)
    def tanh(self):               return Tanh()(self)
    def reshape(self, shape):     return Reshape(shape)(self)
    def flatten(self):            return Reshape((np.prod(np.array(self.shape)),))(self)
    def squeeze(self):            return Reshape(tuple([s for s in self.shape if s>1]))(self)
    def unsqueeze(self, dim):     shape = list(self.shape); shape.insert(dim, 1); return Reshape(tuple(shape))(self)
    def permute(self, dim_ord):   return Permute()(self, dim_ord)
    def T(self):                  return Permute()(self, (1,0))
    def detach(self):             self.op = self.parents = None

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
    def fx(self, *args: Union[Tensor, List[Tensor]]) -> np.ndarray:
        """ Forward function """
        ...

    @abstractmethod
    def vjp(self, y: Tensor, *args: Tensor) -> List[np.ndarray]:
        """
        Vector-Jacobian product.
        Implicitly computes J.T-dot-grad without having to materialize the massive J.
        
        Args:
            y: result tensor of this op (i.e. of the fx function)
            *args: argument tensors to this op, e.g. [x1, x2]
        Returns:
            grads of this op's argument tensors, e.g. [x1_grad, x2_grad]
        """
        ...

# ---
# Operator classes

# Point-wise unary ops

class Neg(Operator):
    def fx(self, x):     return -x.data
    def vjp(self, y, x): return [y.grad * (-1.)]

class Log(Operator):
    def fx(self, x):     return np.log(x.data)
    def vjp(self, y, x): return [y.grad * (1. / (x.data + 1e-8))]

class ReLU(Operator):
    def fx(self, x):     return x.data * (x.data > 0)
    def vjp(self, y, x): return [y.grad * (x.data > 0.).astype(float)]

class LeakyReLU(Operator):
    def __init__(self, alpha=0.2): self.alpha = alpha
    def fx(self, x):               return x.data * ((x.data > 0) + self.alpha * (x.data < 0))
    def vjp(self, y, x):           return [y.grad * ((x.data > 0) + self.alpha * (x.data < 0))]

class Sigmoid(Operator):
    def _sigma(self, x): return 1. / (1. + np.exp(-x))
    def fx(self, x):     return self._sigma(x.data)
    def vjp(self, y, x): return [y.grad * self._sigma(x.data) * (1. - self._sigma(x.data))]

class Tanh(Operator):
    def fx(self, x):     return np.tanh(x.data)
    def vjp(self, y, x): return [y.grad * (1. - np.tanh(x.data)**2)]

# Point-wise binary ops

class Add(Operator):
    def fx(self, x1, x2):     return x1.data + x2.data
    def vjp(self, y, x1, x2): return [y.grad, y.grad]

class Sub(Operator):
    def fx(self, x1, x2):     return x1.data - x2.data
    def vjp(self, y, x1, x2): return [y.grad, -y.grad]

class Mul(Operator):
    def fx(self, x1, x2):     return x1.data * x2.data
    def vjp(self, y, x1, x2): return [y.grad * x2.data, y.grad * x1.data]

class Div(Operator):
    def fx(self, x1, x2):     return x1.data / x2.data
    def vjp(self, y, x1, x2): return [y.grad * (1. / x2.data), y.grad * x1.data * (-1. / (x2.data**2))]

class Pow(Operator):
    def fx(self, x1, x2):     return x1.data**x2.data
    def vjp(self, y, x1, x2):
        x1_grad = y.grad * x2.data * x1.data**(x2.data - 1.)
        if x2.requires_grad: x2_grad = y.grad * np.log(x1.data) * x1.data**x2.data
        else:                x2_grad = np.zeros_like(x2.data)
        return [x1_grad, x2_grad]

# Shape-altering unary ops

class Sum(Operator):
    def __init__(self, dim=None): self.dim = tuple(dim) if isinstance(dim, list) else dim
    def fx(self, x):              return x.data.sum().reshape((1,)) if self.dim is None else x.data.sum(axis=self.dim)
    def vjp(self, y, x):          
        if self.dim is None:            x_grad = np.full(x.shape, y.grad * 1.)
        elif isinstance(self.dim, int): x_grad = np.stack([y.grad]*x.shape[self.dim], axis=self.dim)
        elif isinstance(self.dim, tuple) or isinstance(self.dim, list):
            x_grad = y.grad
            for d in self.dim: x_grad = np.stack([x_grad]*x.shape[d], axis=d)
        return x_grad

class AvgPool1D(Operator):
    # TODO
    def __init__(self, kernel_size, stride, padding): pass
    def fx(self, x):         pass
    def vjp(self, y, x):     pass

class AvgPool2D(Operator):
    # TODO
    def __init__(self, kernel_size, stride, padding): pass
    def fx(self, x):         pass
    def vjp(self, y, x):     pass

# Shape-altering binary ops

class Dot(Operator):
    def fx(self, x1, x2):     return x1.data @ x2.data
    def vjp(self, y, x1, x2): return [y.grad @ x2.data.T, x1.data.T @ y.grad]

class Linear(Operator):  # TODO: Dot() with batch support and bias
    def fx(self, x1, x2):     pass
    def vjp(self, y, x1, x2): pass

class Conv1D(Operator):
    def __init__(self, stride=1, padding=0):
        self.kernel_size, self.stride, self.padding = None, stride, padding # TODO: implement stride, padding, dilation, bias
    def fx(self, x, kernel):
        # assert len(x.shape) == 3 and len(kernel) == 3  # TODO: implement multichannel conv with batch support
        self.kernel_size = kernel.shape[0]
        y = np.zeros(x.shape[0] - int(self.kernel_size-1))
        for i in range(y.shape[0]):
            y[i] = np.sum(kernel.data * x.data[i : i + self.kernel_size])
        return y
    def vjp(self, y, x, kernel):
        x_grad, kernel_grad = np.zeros_like(x.data), np.zeros_like(kernel.data)
        for i in range(y.shape[0]):
            x_grad[i : i + self.kernel_size] += y.grad[i] * kernel.data
            kernel_grad += y.grad[i] * x.data[i : i + self.kernel_size]
        return [x_grad, kernel_grad]

class Conv2D(Operator):
    # TODO
    def __init__(self, stride=1, padding=0): pass
    def fx(self, x, kernel):         pass
    def vjp(self, y, x, kernel):     pass

# Shape transformation ops

class Slice(Operator):
    def __init__(self, idx): self.idx = idx
    def fx(self, x):         return x.data[self.idx]
    def vjp(self, y, x):
        x_grad = np.zeros_like(x.data)
        x_grad[self.idx] = y.grad
        return [x_grad]

class Reshape(Operator):
    def __init__(self, shape): self.shape = shape
    def fx(self, x):           return x.data.reshape(self.shape)
    def vjp(self, y, x):       return [y.grad.reshape(x.shape)]
    
class Permute(Operator):
    def __init__(self, dim_ord): self.dim_ord = dim_ord
    def fx(self, x):             return x.data.transpose(self.dim_ord)
    def vjp(self, y, x):         return [y.grad.transpose(self.dim_ord)]
    
class Stack(Operator):
    def __init__(self, dim):   self.dim = dim
    def fx(self, x_list):      return np.stack([x.data for x in x_list], axis=self.dim)
    def vjp(self, y, *x_list):
        x_grad_list = np.split(y.grad, len(x_list), axis=self.dim)
        return [x_grad.squeeze(axis=self.dim) for x_grad in x_grad_list]

class Cat(Operator):
    def __init__(self, dim):   self.dim = dim
    def fx(self, x_list):      return np.concatenate([x.data for x in x_list], axis=self.dim)
    def vjp(self, y, *x_list): 
        x_grad_list = np.split(y.grad, [x.shape[self.dim] for x in x_list], axis=self.dim)
        return x_grad_list

# ---
# Convenience functions

def zeros(shape, requires_grad=False):               return Tensor(np.zeros(shape), requires_grad=requires_grad)
def ones(shape, requires_grad=False):                return Tensor(np.ones(shape), requires_grad=requires_grad)
def rand(shape, requires_grad=False):                return Tensor(np.random.rand(size=shape), requires_grad=requires_grad)
def randn(shape, requires_grad=False):               return Tensor(np.random.normal(size=shape), requires_grad=requires_grad)
def randint(start, end, shape, requires_grad=False): return Tensor(np.random.randint(start, end, shape), requires_grad=requires_grad)
def stack(x_list, dim):                              return Stack(dim)(x_list)
def cat(x_list, dim):                                return Cat(dim)(x_list)
def repeat(x, repeats, dim):                         return Cat(dim)([x for _ in range(repeats)])
def conv1d(x, kernel, stride=1, padding=0):          return Conv1D(stride, padding)(x, kernel)

# ---
# Internal utils

def _generate_tensor_name():
    global _tensor_namespace
    if len(_tensor_namespace) == 0: tensor_name = '0'
    else:                           tensor_name = str(int(_tensor_namespace[-1]) + 1)
    _tensor_namespace.append(tensor_name)
    return tensor_name
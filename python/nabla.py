from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import graphviz
import matplotlib.pyplot as plt

_grad_enabled = True  # Global switch
_tensor_namespace = [] # Global list of all tensor names


# ---
# Core functions

def enable_grad(flag):
    global _grad_enabled
    _grad_enabled = flag

def show_dag(tensor, view_img=True):
    graph = graphviz.Digraph(f"DAG_TensorID={tensor.name}", format='png')
    def traverse_dag(tnsr):  # Build graph with recursive depth-first tree traversal
        if tnsr.parents is not None:
            for i in range(len(tnsr.parents)):
                from_node = f"{(tnsr.parents[i].op.__class__.__name__)} {tnsr.parents[i].name}"
                to_node = f"{tnsr.op.__class__.__name__} {tnsr.name}"
                graph.edge(from_node, to_node, tnsr.parents[i].name)
                traverse_dag(tnsr.parents[i])
    traverse_dag(tensor)    
    if view_img:
        graph.render(directory='/tmp', view=False)
        img = plt.imread(f"/tmp/DAG_TensorID={tensor.name}.gv.png")  
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
    
    def backward(self, grad: np.ndarray = np.array([[1.]])): # Backprop function
        if not _grad_enabled: raise RuntimeError("Called backward() but gradient computation is disabled.")
        self._accumulate_grad(grad)
        if self.op is not None:
            if isinstance(self.op, Stack) or isinstance(self.op, Cat): 
                parents = self.parents[0]
            else:
                parents = self.parents
            parent_vjps = self.op.vjp(self, *parents)
            for i in range(len(parents)):
                if parents[i].requires_grad:
                    parent_grad = parent_vjps[i]
                    parents[i].backward(parent_grad)  # Recursive depth-first tree traversal

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
    def slice(self, dims, ranges): return Slice()(self, dims, ranges)
    def dot(self, other):         return Dot()(self, other)
    def sum(self):                return Sum()(self)
    def mean(self):               return Sum()(self) / Tensor(np.prod(np.array(self.shape)))
    def log(self):                return Log()(self)
    def relu(self):               return ReLU()(self)
    def leaky_relu(self, alpha):  return LeakyReLU(alpha)(self)
    def sigmoid(self):            return Sigmoid()(self)
    def tanh(self):               return Tanh()(self)
    def reshape(self, shape):     return Reshape()(self, shape)
    def flatten(self):            return Reshape()(self, (np.prod(np.array(self.shape)),))
    def squeeze(self):            return Reshape()(self, tuple([s for s in self.shape if s>0]))
    def unsqueeze(self, dim):     return Reshape()(self, tuple(list(self.shape).insert(dim, 1)))
    def permute(self, dim_ord):   return Permute()(self, dim_ord)
    def T(self):                  return Permute()(self, (1,0))

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
# Shape transformation ops

class Slice(Operator):
    # TODO: Make API fully numpy-like
    def __init__(self, dims, ranges):
        self.dims = dims
        self.ranges = ranges
        nplike_slice_idx = []
        for i in range(len(x.shape)):
            if i in self.dims:
                range = self.ranges[self.dims.index(i)]
                nplike_slice_idx.append(f"{range[0]}:{range[1]}")
            else:
                nplike_slice_idx.append(":")
        self.nplike_slice_idx = ','.join(nplike_slice_idx)
    def fx(self, x): return eval(f"x[{self.nplike_slice_idx}]")
    def vjp(self, y, x):
        x_grad = np.zeros_like(x.data)
        eval(f"x_grad[{self.nplike_slice_idx}] = y.grad")
        return [x_grad]

class Reshape(Operator):
    def fx(self, x, shape): return x.data.reshape(shape)
    def vjp(self, y, x):
        x_grad = y.grad.reshape(x.shape)
        return [x_grad]
    
class Permute(Operator):
    def __init__(self, dim_ord): self.dim_ord = dim_ord
    def fx(self, x):             return x.data.transpose(self.dim_ord)
    def vjp(self, y, x):
        x_grad = y.grad.transpose(self.dim_ord)
        return [x_grad]
    
class Stack(Operator):
    def __init__(self, dim):
        self.dim = dim
        self.x_dimsize_list = None
    def fx(self, x_list): 
        self.x_dimsize_list = [x.shape[self.dim] for x in x_list]
        return np.stack(x_list, axis=self.dim)
    def vjp(self, y, x_list):
        x_grad_list = np.array_split(y.grad, self.x_dimsize_list[:-1], axis=self.dim)
        return [x_grad.squeeze() for x_grad in x_grad_list]

class Cat(Operator):
    def __init__(self, dim):
        self.dim = dim
        self.x_dimsize_list = None
    def fx(self, x_list): 
        self.x_dimsize_list = [x.shape[self.dim] for x in x_list]
        return np.concatenate(x_list, axis=self.dim)
    def vjp(self, y, x_list):
        x_grad_list = np.array_split(y.grad, self.x_dimsize_list[:-1], axis=self.dim)
        return x_grad_list
    

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

def stack(x_list, dim):
    return Stack(dim)(x_list)

def cat(x_list, dim):
    return Cat(dim)(x_list)

def repeat(x, repeats, dim):
    return Cat(dim)([x for _ in range(repeats)])

# ---
# Internal utils

def _generate_tensor_name():
    if len(_tensor_namespace) == 0:
        tensor_name = '0'
    else:
        tensor_name = str(int(_tensor_namespace[-1]) + 1)
    _tensor_namespace.append(tensor_name)
    return tensor_name
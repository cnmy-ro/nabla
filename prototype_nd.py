import numpy as np


class Variable:
    
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad = np.zeros_like(data)  # droot / dself
        self.shape = data.shape
        self.prev = None
    
    def backward(self, grad: np.ndarray = 1.0) -> None:

        self.grad += grad
        
        # Depth-first traversal
        if self.prev is not None:

            fn, fn_vars = self.prev[0], self.prev[1]
            fn_vars_vjp = fn.vjp(grad, self, *fn_vars)
            
            for i in range(len(fn_vars)):
                fn_var_grad = fn_vars_vjp[i]
                fn_vars[i].backward(fn_var_grad)

    def __neg__(self):
        return Neg()(self)

    def __add__(self, other):
        return Add()(self, other)

    def __sub__(self, other):
        return Sub()(self, other)

    def __mul__(self, other):
        return Mul()(self, other)

    def __truediv__(self, other):
        return Div()(self, other)

    def __pow__(self, other):
        return Pow()(self, other)

    def sum(self):
        return Sum()(self)

    def relu(self):
        return ReLU()(self)

    def tanh(self):
        return Tanh()(self)

    def dot(self, other):
        return Dot()(self, other)


class Function:
    
    def __call__(self, *args: Variable) -> Variable:
        y = self.fx(*args)
        y.prev = [self, args]
        return y

    def fx(self, *args: Variable) -> Variable:
        """ Forward function """
        ...

    def vjp(self, grad: np.ndarray, result: Variable, *args: Variable) -> list[np.ndarray]:
        """
        Vector-Jacobian product
        Implicitly computes J-dot-grad without having to materialize the massive J
        """
        ...


class Neg(Function):

    def fx(self, x):
        return Variable(-x.data)

    def vjp(self, grad, result, x):
        return [-grad]


class Sum(Function):

    def fx(self, x):
        return Variable(x.data.sum())

    def vjp(self, grad, result, x):
        return [grad]


class ReLU(Function):

    def fx(self, x):
        return Variable(np.maximum(x.data, np.zeros_like(x.data)))

    def vjp(self, grad, result, x):
        return [grad * (result > 0).astype(float)]


class Tanh(Function):

    def fx(self, x):
        return Variable(np.tanh(x.data))

    def vjp(self, grad, result, x):
        return [grad * (1 - np.tanh(x.data)**2)]


class Add(Function):

    def fx(self, x1, x2):
        return Variable(x1.data + x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad, grad]


class Sub(Function):

    def fx(self, x1, x2):
        return Variable(x1.data - x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad, -grad]


class Mul(Function):

    def fx(self, x1, x2):
        return Variable(x1.data * x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad * x2.data, grad * x1.data]


class Div(Function):

    def fx(self, x1, x2):
        return Variable(x1.data / x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad * (1.0 / x2.data), grad * x1.data * (-1.0 / (x2.data**2))]


class Pow(Function):

    def fx(self, x1, x2):
        return Variable(x1.data ** x2)

    def vjp(self, grad, result, x1, x2):
        return [grad * x2 * x1.data ** (x2 - 1)]


class Dot(Function):

    def fx(self, x1, x2):
        return Variable(x1.data @ x2.data)

    def vjp(self, grad, result, x1, x2):
        return [None, None]



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    def run_demo():

        """ App demonstrating the universal approximation capability of an NN. """
        
        xtrain, ytrain = None
        xtest, ytest = None

        def mlp(x, params):
            z1 = params['w1'].dot(x)
            a1 = x.relu()
            z2 = params['w2'].dot(a1)
            y = z2.tanh()            
            return y

        def init_params():
            params = {'w1': None, 'w2': None}
            return params

        def update_params(params, lr):
            params = {k: v - lr*v.grad for k,v in params.items()}
            return params

        def zero_grad(params):
            params = {k: v.grad = np.zeros_like(v.data) for k,v in params.items()}
            return params

        for e in range(1, 10):
                        
            for i in range(xtrain.shape[0]):
                
                x_i = Variable(xtrain[i])
                ytrue_i = Variable(ytrain[i])
                ypred_i = mlp(x_i, params)

                loss = (ypred_i - ytrue_i) ** 2
                loss.backward()
                params = update_params(params)
                params = zero_grad(params)

            # Viz
            # ...
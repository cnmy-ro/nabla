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
            fn_vars_vjp = fn.vjp(self.grad, self, *fn_vars)
            
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

    def dot(self, other):
        return Dot()(self, other)

    def sum(self):
        return Sum()(self)

    def relu(self):
        return ReLU()(self)

    def sigmoid(self):
        return Sigmoid()(self)

    def tanh(self):
        return Tanh()(self)


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
        Vector-Jacobian product.
        Implicitly computes J-dot-grad without having to materialize the massive J.
        """
        ...


class Neg(Function):

    def fx(self, x):
        return Variable(-x.data)

    def vjp(self, grad, result, x):
        return [-grad]


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
        return [grad * (1. / x2.data), grad * x1.data * (-1. / (x2.data**2))]


class Pow(Function):

    def fx(self, x1, x2):
        return Variable(x1.data ** x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad * x2.data * x1.data ** (x2.data - 1.), grad * np.log(x1.data) * x1.data ** x2.data]


class Dot(Function):

    def fx(self, x1, x2):
        return Variable(x1.data @ x2.data)

    def vjp(self, grad, result, x1, x2):
        return [grad @ x2.data.T, x1.data.T @ grad]


class Sum(Function):

    def fx(self, x):
        return Variable(x.data.sum())

    def vjp(self, grad, result, x):
        return [grad]


class ReLU(Function):

    def fx(self, x):
        return Variable(np.maximum(x.data, np.zeros_like(x.data)))

    def vjp(self, grad, result, x):
        return [grad * (result.data > 0.0).astype(float)]


class Sigmoid(Function):

    def _sigma(self, x):
        return 1. / (1. + np.exp(-x))

    def fx(self, x):
        return Variable(self._sigma(x.data))

    def vjp(self, grad, result, x):
        return [grad * self._sigma(x.data) * (1. - self._sigma(x.data))]


class Tanh(Function):

    def fx(self, x):
        return Variable(np.tanh(x.data))

    def vjp(self, grad, result, x):
        return [grad * (1. - np.tanh(x.data)**2)]



if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    def run_demo():

        """ App demonstrating the universal approximation capability of an NN. """
        
        xtrain = np.random.rand(1000) * 2 * np.pi
        noise = np.random.normal(0, 0.01, size=(1000,))
        ytrain = np.sin(xtrain) + noise

        xtest = np.linspace(0, 2*np.pi, 1000)
        ytest = np.sin(xtest)

        # Dummy value "1" to use with learnable bias
        xtrain = np.stack((xtrain, np.ones_like(xtrain)), axis=1)
        xtest = np.stack((xtest, np.ones_like(xtest)), axis=1)

        def mlp(x, params):
            z1 = params['w1'].dot(x)
            a1 = z1.sigmoid()
            z2 = params['w2'].dot(a1)
            y = z2.tanh()          
            return y

        def init_params():
            params = {
            'w1': Variable(np.random.normal(size=(8, 2))),
            'w2': Variable(np.random.normal(size=(1, 8)))
            }
            return params

        def update_params(params, lr):
            for k in params.keys():
                params[k].data = params[k].data - lr*params[k].grad
            return params

        def zero_grad(params):
            for k in params.keys():
                params[k].grad = np.zeros_like(params[k].data)
            return params

        def test(params, xtest, ytest):
            ytestpred = []
            for i in range(xtest.shape[0]):
                x_i = np.expand_dims(xtest[i] / np.pi - 1., axis=1)
                z1 = np.dot(params['w1'].data, x_i)
                a1 = 1 / (1 + np.exp(-z1))
                z2 = np.dot(params['w2'].data, a1)
                ypred_i = np.tanh(z2)
                ytestpred.append(np.squeeze(ypred_i))
            ytestpred = np.array(ytestpred)

            return ytestpred
    
        params = init_params()
        ytestpred = test(params, xtest, ytest)
        
        fig, ax = plt.subplots()
        testgt_plot = plt.plot(xtest[:,0], ytest, c='tab:blue')[0]
        testpred_plot = plt.plot(xtest[:,0], ytestpred, c='tab:orange')[0]
        plt.ion()
        plt.show()

        for e in tqdm(range(1, 50)):

            for i in range(xtrain.shape[0]):
                
                x_i = Variable(np.expand_dims(xtrain[i] / np.pi - 1., axis=1))
                ytrue_i = Variable(np.expand_dims(np.expand_dims(ytrain[i], axis=0), axis=0))
                ypred_i = mlp(x_i, params)

                loss = (ypred_i - ytrue_i) ** Variable(np.array([[2.]]))
                loss.backward()
                params = update_params(params, lr=0.5)
                params = zero_grad(params)                        

            ytestpred = test(params, xtest, ytest)
            testpred_plot.set_ydata(ytestpred)
            fig.canvas.draw()
            time.sleep(0.005)
            fig.canvas.flush_events()    


    run_demo()
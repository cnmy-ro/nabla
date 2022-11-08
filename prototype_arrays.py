from abc import ABC, abstractmethod
import numpy as np



class Variable:
    
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad = np.zeros_like(data)  # droot / dself
        self.shape = data.shape
        self.is_leaf = True
        self.prev = None
    
    def backward(self, grad: np.ndarray = 1.0) -> None:

        self.grad += grad
        
        # Depth-first traversal
        if not self.is_leaf:
            op, op_vars = self.prev[0], self.prev[1]
            op_vars_local_deriv = op.derivative(*op_vars)
            
            for i in range(len(op_vars)):

                # Chain rule
                if len(self.shape) >= 1:
                    op_var_grad = op_vars_local_deriv[i] @ self.grad 
                else:
                    op_var_grad = op_vars_local_deriv[i] * self.grad 
                
                # print(op_vars_local_deriv[i], self.grad)
                op_vars[i].backward(op_var_grad)


    def __neg__(self):
        return Negate()(self)

    def __add__(self, x2):
        return Add()(self, x2)

    def __mul__(self, x2):
        return Multiply()(self, x2)


class UnaryOperator(ABC):
    
    def __call__(self, x: Variable) -> Variable:
        y_data = self.forward(x)
        y = Variable(y_data)
        y.is_leaf = False
        y.prev = (self, [x])
        return y

    @abstractmethod
    def forward(self, x: Variable) -> np.ndarray:
        ...

    @abstractmethod
    def derivative(self, x: Variable) -> list[np.ndarray]:
        ...


class BinaryOperator(ABC):
    
    def __call__(self, x1: Variable, x2: Variable) -> Variable:
        y_data = self.forward(x1, x2)
        y = Variable(y_data)
        y.is_leaf = False
        y.prev = [self, [x1, x2]]
        return y

    @abstractmethod
    def forward(self, x1: Variable, x2: Variable) -> np.ndarray:
        ...

    @abstractmethod
    def derivative(self, x1: Variable, x2: Variable) -> list[np.ndarray, np.ndarray]:
        ...


class Negate(UnaryOperator):

    def forward(self, x: Variable) -> np.ndarray:
        return -x.data

    def derivative(self, x: Variable) -> list[np.ndarray]:
        dx = np.eye(x.data.flatten().shape[0]) * (-1)
        dx = dx.reshape(x.shape + x.shape)
        return dx


class Sum(UnaryOperator):

    def forward(self, x: Variable) -> np.ndarray:
        return x.data.sum()

    def derivative(self, x: Variable) -> list[np.ndarray]:
        dx = np.ones_like(x.data)
        return dx


class Add(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> np.ndarray:
        return x1.data + x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[np.ndarray, np.ndarray]:
        dx1 = np.eye(x1.data.flatten().shape[0])
        dx1 = dx1.reshape(x1.shape + x1.shape)

        dx2 = np.eye(x2.data.flatten().shape[0])
        dx2 = dx2.reshape(x2.shape + x2.shape)

        return [dx1, dx2]


class Multiply(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> np.ndarray:
        return x1.data * x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[np.ndarray, np.ndarray]:

        dx1 = np.diag(x2.data.flatten())
        dx1 = dx1.reshape(x2.shape + x1.shape)

        dx2 = np.diag(x1.data.flatten())
        dx2 = dx2.reshape(x1.shape + x2.shape)

        return [dx1, dx2]







if __name__ == '__main__':

    
    a = Variable(np.array([1., 1.]))
    b = Variable(np.array([3., 3.]))
     
    c = a * b
    o = Sum()(c)
    print(o.data)

    o.backward()
    print(o.grad)
    print(c.grad)
    print(a.grad)
    print(b.grad)
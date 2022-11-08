from abc import ABC, abstractmethod



class Variable:
    
    def __init__(self, data: float) -> None:
        self.data = data
        self.grad = 0.0  # droot / dself
        self.is_leaf = True
        self.prev = None
    
    def backward(self, grad: float = 1.0) -> None:

        self.grad += grad
        
        # Depth-first traversal
        if not self.is_leaf:
            op, op_vars = self.prev[0], self.prev[1]
            op_vars_local_deriv = op.derivative(*op_vars)
            
            for i in range(len(op_vars)):
                op_var_grad = self.grad * op_vars_local_deriv[i]  # Chain rule
                op_vars[i].backward(op_var_grad)


    def __neg__(self):
        return Negate()(self)

    def __add__(self, x2):
        return Add()(self, x2)

    def __sub__(self, x2):
        return Subtract()(self, x2)

    def __mul__(self, x2):
        return Multiply()(self, x2)

    def __truediv__(self, x2):
        return Divide()(self, x2)


class UnaryOperator(ABC):
    
    def __call__(self, x: Variable) -> Variable:
        y_data = self.forward(x)
        y = Variable(y_data)
        y.is_leaf = False
        y.prev = [self, [x]]
        return y

    @abstractmethod
    def forward(self, x: Variable) -> float:
        ...

    @abstractmethod
    def derivative(self, x: Variable) -> list[float]:
        ...


class BinaryOperator(ABC):
    
    def __call__(self, x1: Variable, x2: Variable) -> Variable:
        y_data = self.forward(x1, x2)
        y = Variable(y_data)
        y.is_leaf = False
        y.prev = [self, [x1, x2]]
        return y

    @abstractmethod
    def forward(self, x1: Variable, x2: Variable) -> float:
        ...

    @abstractmethod
    def derivative(self, x1: Variable, x2: Variable) -> list[float, float]:
        ...


class Negate(UnaryOperator):

    def forward(self, x: Variable) -> float:
        return -x.data

    def derivative(self, x: Variable) -> list[float]:
        return [-1.0]


class Add(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> float:
        return x1.data + x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[float, float]:
        return [1.0, 1.0]


class Subtract(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> float:
        return x1.data - x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[float, float]: 
        return [1.0, -1.0]


class Multiply(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> float:
        return x1.data * x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[float, float]:
        return [x2.data, x1.data]


class Divide(BinaryOperator):

    def forward(self, x1: Variable, x2: Variable) -> float:
        return x1.data / x2.data

    def derivative(self, x1: Variable, x2: Variable) -> list[float, float]:
        return [1 / x2.data, - 1 / (x1.data ** 2)]
    




if __name__ == '__main__':

    
    a = Variable(3.0)
    b = Variable(2.0)
    c = Variable(4.0)
     
    o = a*a + a + b*b - b
    print(o.data)

    o.backward()
    print(o.grad)
    print(a.grad)
    print(b.grad)
    # print(c.grad)
    # print(d.grad)

    a.grad = 0.0
    b.grad = 0.0
    o = a / b
    print(o)
    o.backward()
    print(o.grad)
    print(a.grad)
    print(b.grad)
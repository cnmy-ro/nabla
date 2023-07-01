from abc import ABC, abstractmethod



class Variable:
    
    def __init__(self, data: float):
        self.data = data
        self.grad = 0.0  # droot / dself
        self.is_leaf = True
        self.prev = None
    
    def backward(self, grad: float = 1.0):
        self.grad += grad        
        if not self.is_leaf: # Depth-first tree walk
            fn, fn_vars = self.prev[0], self.prev[1]
            fn_vars_local_deriv = fn.dfx(*fn_vars)            
            for i in range(len(fn_vars)):
                fn_var_grad = self.grad * fn_vars_local_deriv[i]  # Chain rule
                fn_vars[i].backward(fn_var_grad)

    def __neg__(self): return Neg()(self)
    def __add__(self, x2): return Add()(self, x2)
    def __sub__(self, x2): return Sub()(self, x2)
    def __mul__(self, x2): return Mul()(self, x2)
    def __truediv__(self, x2): return Div()(self, x2)

class Operator(ABC):
    
    def __call__(self, *args: Variable) -> Variable:
        y_data = self.fx(*args)
        y = Variable(y_data)
        y.is_leaf = False
        y.prev = [self, args]
        return y

    @abstractmethod
    def fx(self, *args: Variable) -> float:
        ...

    @abstractmethod
    def dfx(self, *args: Variable) -> list[float]:
        ...

class Neg(Operator):
    def fx(self, x): return -x.data
    def dfx(self, x): return [-1.0]

class Add(Operator):
    def fx(self, x1, x2): return x1.data + x2.data
    def dfx(self, x1, x2): return [1.0, 1.0]

class Sub(Operator):
    def fx(self, x1, x2): return x1.data - x2.data
    def dfx(self, x1, x2): return [1.0, -1.0]

class Mul(Operator):
    def fx(self, x1, x2): return x1.data * x2.data
    def dfx(self, x1, x2): return [x2.data, x1.data]

class Div(Operator):
    def fx(self, x1, x2): return x1.data / x2.data
    def dfx(self, x1, x2): return [1.0 / x2.data, (-1.0 / (x2.data ** 2) * x1.data)]
    



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
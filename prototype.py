

class Var:
    
    def __init__(self, data: float):
        self.data = data
        self.grad = 0.0  # droot / dself
        self.is_leaf = True
        self.prev = []
    
    def backward(self):

        self.grad = 1.0
        
        # TODO: Do breadth-first graph traversal
        for i in range(len(self.prev)):
            prev_op = self.prev[i][0]
            op_vars = self.prev[i][1]
            op_vars_grads = prev_op.derivative(*op_vars)
            
            for j in range(len(op_vars)):
                op_vars[j].grad = op_vars_grads[j]
            

class Multiply:

    def __call__(self, x1: Var, x2: Var) -> Var:
        y = Var(x1.data * x2.data)
        y.is_leaf = False
        y.prev.append([self, [x1, x2]])
        return y

    def derivative(self, x1: Var, x2: Var) -> list[float, float]:
        return [x2.data, x1.data]
    

class Add:

    def __call__(self, x1: Var, x2: Var) -> Var:
        y = Var(x1.data + x2.data)
        y.is_leaf = False
        y.prev.append([self, [x1, x2]])
        return y

    def derivative(self, x1: Var, x2: Var) -> list[float, float]:
        return [1.0, 1.0]




if __name__ == '__main__':

    
    a = Var(3.0)
    b = Var(2.0)
    mul_op = Multiply()
    
    c = mul_op(a, b)
    print(c.data)

    c.backward()
    print(c.grad)
    print(a.grad)
    print(b.grad)


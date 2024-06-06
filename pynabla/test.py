import numpy as np
import nabla
from nabla import Tensor


def mul():
	a = Tensor(np.ones((3,3,3))*2, requires_grad=True)
	b = Tensor(np.ones((3,3,3))*4, requires_grad=True)
	c = a * b
	l = c.sum()
	l.backward()
	print(c.grad)
	print(a.grad)
	print(b.grad)

def test_dot():
	a = Tensor(np.ones((3,4))*2, requires_grad=True)
	b = Tensor(np.ones((4,2))*4, requires_grad=True)
	c = a.dot(b)
	l = c.sum()
	l.backward()
	print(c.grad)
	print(a.grad)
	print(b.grad)


test_dot()
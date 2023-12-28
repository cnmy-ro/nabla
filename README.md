<p align="center">
  <img src="https://raw.githubusercontent.com/cnmy-ro/nabla/main/docs/logo.png">
</p>


# Nabla

Minimal implementation of reverse-mode automatic differentiation.

- Python version:
	- `nabla_python/nabla.py`: Core autodiff module	based on Numpy
	- `examples`: Toy examples built using this module
- C version:
	- `nabla_c/arrays.h`: Custom array library
	- `nabla_c/nabla.h`: Core autodiff library wrapping `arrays.h`
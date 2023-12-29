<p align="center">
  <img src="https://raw.githubusercontent.com/cnmy-ro/nabla/main/docs/logo.png">
</p>


# Nabla

Minimal implementation of reverse-mode automatic differentiation with PyTorch-like API.

- Python version:
	- `pynabla/nabla.py`: Core autodiff module	based on Numpy
	- `examples`: Toy examples built using this module
- C version:
	- `cnabla/arrays.h`: Custom array library
	- `cnabla/nabla.h`: Core autodiff library wrapping `arrays.h`
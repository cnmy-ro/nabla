<p align="center">
  <img src="https://raw.githubusercontent.com/cnmy-ro/nabla/main/docs/logo2.png">
</p>


# Nabla

Minimal implementation of reverse-mode automatic differentiation.

- Python version:
	- `python/nabla.py`: Thin autodiff wrapper over Numpy with PyTorch-like API
	- `python_examples`: Toy examples built using this module
- C version:
	- `c/cpuarrays.h` : Low-level array library for CPU
	- `c/nabla.h`: Autodiff library wrapping `cpuarrays`
	- `c_examples`: Toy examples built using this library

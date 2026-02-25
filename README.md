<p align="center">
  <img src="docs/logo.png"  width="300">
</p>


Nabla is minimal implementation of reverse-mode automatic differentiation.

- Python version:
	- `python/nabla.py`: Thin autodiff wrapper over Numpy with PyTorch-like API
	- `python_examples`: Toy examples built using this module
- C version (WIP):
	- `c/cpuarrays.h` : Low-level array library for CPU
	- `c/nabla.h`: Autodiff library wrapping `cpuarrays`
	- `c_examples`: Toy examples built using this library

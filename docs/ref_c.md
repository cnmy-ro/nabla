# Reference for the Nabla C Library

## Design

The modular design of the Nabla C library separates the array computation logic from the autodiff logic.


### `cpuarrays.h`  (WIP)
Low-level array library. Implements array operations as loops that run on the CPU. Mimics the functionality of Numpy. For memory safety, arrays need to be explicitly malloc'd before performing most operations (with all shape ops as exception). Thus, operations are viewed as data transformations of already allocated memory. The responsibility of freeing the memory is also handed to the user.

All functions are call-by-reference to avoid overhead in copying arrays in arguments and return values.


### `nabla.h` (WIP)
Higher-level library that wraps `cpuarrays.h` with autodiff logic. Since the implementation of low-level array ops is abstracted away, perhaps one could easily swap `cpuarrays.h` with a GPU-based parallelized alternative while fully retaining the `nabla.h` autodiff code. Also takes care of array memory management.
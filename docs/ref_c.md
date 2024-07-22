# Reference for the Nabla C Library

## Design

### `cpuarrays.h`  (WIP)
Low-level array library. Implements array operations as loops that run on the CPU. Mimics the funcationality of Numpy. For memory safety, arrays need to be explicitly malloc'd before performing most operations. Thus, operations are viewed as data transformations of already allocated memory. The responsibility of freeing the memory is handed to the user.


### `nabla.h` (WIP)
Library that wraps `cpuarrays` with autodiff logic. Since the implementation of low-level array ops is abstracted away, perhaps one could easily swap `cpuarrays` with a powerful GPU-based alternative while retaining the autodiff code. Also, takes care of array memory management.
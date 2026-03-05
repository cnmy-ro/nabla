"""
Nabla functional form
"""

# nabla core module:
# - Nabla differential functions:
#   - grad, vjp, jvp, jac, hess, divergence, curl, laplacian
# - Tensor class
# - Tensor operator functions
#   - Arithmetic ops: neg, add, sub, mul, div, exp, log
#   - Shapeshift ops: sum, prod, vecdot, matmul, squeeze, unsqueeze, stack, repeat, cat, permute
#   - Indexing ops: index, slice, where, argwhere
#   - Misc math ops: thresholding, etc.
# - Convenience funcs:
#   - Zeros, ones, rand, randn, randint,  

# nabla.nn module:
# - Batched ops for fixed tensor shapes: linear, conv, attention
# - Activation functions: relu, leakyrelu, sigmoid, tanh
# - Optimizers: GD, Adam

# nabla.linalg module:
# - Matrix inversion: inv, pinv
# - Decompositions: ED, SVD, Cholesky, LU, QR

# nabla.phy module:  for physics and imaging
# - FFT
# - Reprs (grid-based or continuous): measurement, latents (image, etc.)
# - Forward ops
# - Inversion routines
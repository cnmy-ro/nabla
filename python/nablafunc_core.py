"""
Nabla functional form
"""

# nabla core module:
# - Core abstractions: tensor, function (operating on tensors), transform (operating on functions)
# - Tensor class
# - Tensor functions:
#   - Arithmetic: neg, add, sub, mul, div, exp, log
#   - Shapeshift: sum, prod, vecdot, matmul, squeeze, unsqueeze, stack, repeat, cat, permute
#   - Indexing: index, slice, where, argwhere
#   - Misc math: thresholding, etc.
#   - Init: empty, zeros, ones, rand, randn, randint, etc.
# - Differential transforms:
#   - grad, vjp, jvp, jac, hess, divergence, curl, laplacian

# nabla.nn module:
# - Batched functions for fixed tensor shapes: linear, conv, attention
# - Activation functions: relu, leakyrelu, sigmoid, tanh
# - Optimizers: GD, Adam

# nabla.linalg module:
# - Matrix inversion: inv, pinv
# - Decompositions: ED, SVD, Cholesky, LU, QR

# nabla.phy module:  for physics and imaging
# - FFT
# - Reprs (grid-based or continuous): measurement, latents (image, etc.)
# - Forward operators
# - Inversion routines
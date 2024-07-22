/*
Low-level array library for CPU
*/

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef CPUARRAYS_H
#define CPUARRAYS_H


// ---
// Core data structure

struct NDArray {	
	int ndims; // Num elements in shape
	int* shape;
	size_t* stride;
	int numel; // Num elements in arr
	size_t size; // size in bytes = numel * sizeof(float)
	float* arr;
};
typedef struct NDArray NDArray;


// ---
// Core utils

int calc_numel(int ndims, int* shape) {
	int numel = 1;
	for (int d=0; d<ndims; d++)
		numel = numel * (*(shape + d));
	return numel;
}
void malloc_array(NDArray* x, int ndims, int* shape) {
	
	// Init metadata variables
	x->ndims = ndims;
	x->shape = malloc(ndims * sizeof(int));
	for (int d=0; d<ndims; d++)
		*(x->shape + d) = *(shape + d);
	x->stride = malloc(ndims * sizeof(size_t));
	*(x->stride + ndims - 1) = sizeof(float);
	for (int d=ndims-2; d>=0; d--) {
		*(x->stride + d) = *(x->stride + d + 1) * (*(x->shape + d + 1));
	}	
	x->numel = calc_numel(ndims, shape);
	x->size = x->numel * sizeof(float);

	// Dynamically allocate array memory
	printf("malloc %zu\n", x->size);
	x->arr = malloc(x->numel * sizeof(float));
}
void free_array(NDArray* x) {
	free(x->arr);
	free(x->stride);
	free(x->shape);
	x->size = 0;
	x->numel = 0;
	x->ndims = 0;
}
void copy_array(NDArray* x, NDArray* y) {
	for (int idx=0; idx<x->numel; idx++)
		*(y->arr + idx) = *(x->arr + idx);
}
void print_array(NDArray* x) {
	for (int idx=0; idx<x->numel; idx++)
		printf("%f ", *(x->arr + idx));
	// TODO: format properly
}


// ---
// Initialization operators

void init_array_full(NDArray* x, float fill_value) {
	for (int idx=0; idx<x->numel; idx++)
		*(x->arr + idx) = fill_value;
}
void init_array_linspace(NDArray* x, float start, float end, int steps) {
	float alpha;
	for (int idx=0; idx<x->numel; idx++) {
		alpha = (float)idx / (float)steps;
		*(x->arr + idx) = (1 - alpha)*start + alpha*end;
	}
}
void init_array_rand(NDArray* x) {
	srand(time(NULL));
	for (int idx=0; idx<x->numel; idx++)
		*(x->arr + idx) = rand() / (float)RAND_MAX;
}
void init_array_randn(NDArray* x) {
	// TODO
	// Use Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}
void init_array_randint(NDArray* x, float start, float end) {
	srand(time(NULL));
	float randint;
	for (int idx=0; idx<x->numel; idx++) {
		randint = (rand() / (float)RAND_MAX) * (end - start) + start;
		*(x->arr + idx) = rand() / (float)RAND_MAX;
	}
}


// ---
// Shape operators
/* Note: The output arrays are malloc'd inside these functions. */	

// Indexing / slicing
void slice(NDArray* x, NDArray* y, int dim, int idx_along_dim) {
	/*
	Numpy equivalent -- array[:,idx_along_dim,:,:] where the array is 4D amd dim=1
	*/

	// Determine slice
	int dim_len = *(x->shape + dim);
	int dim_stride_int = (int)(*(x->stride + dim) / sizeof(float));
	int seq_idx_start = idx_along_dim * dim_stride_int;
	int seq_idx_end = seq_idx_start + dim_len*dim_stride_int;

	// Malloc output array
	int yndims = x->ndims - 1;
	int* yshape =  malloc(yndims * sizeof(int));	
	int yd = 0;
	for (int xd=0; xd<x->ndims; xd++) {
		if (xd == dim)
			continue;
		*(yshape + yd) = *(x->shape + xd);
		yd++;
	}
	malloc_array(y, yndims, yshape);
	free(yshape);

	// Copy values into output array
	int yidx = 0;
	for (int xidx=seq_idx_start; xidx<seq_idx_end; xidx+=dim_stride_int) {
		*(y->arr + yidx) = *(x->arr + xidx);
		yidx += 1;
	}
}

// Mutation
void squeeze(NDArray* x, NDArray* y) {	

	// Determine squeezed array's ndims and shape
	int* squeezed_dims =  malloc(x->ndims * sizeof(int));
	int squeezed_ndims = 0;
	for (int d=0; d<x->ndims; d++) {
		if (*(x->shape + d) > 1) {
			*(squeezed_dims + squeezed_ndims) = d;
			squeezed_ndims += 1;
		}
	}
	int* squeezed_shape = malloc(squeezed_ndims * sizeof(int));
	int xdim;
	for (int yd=0; yd<squeezed_ndims; yd++) {
		xdim = *(squeezed_dims + yd);
		*(squeezed_shape + yd) = *(x->shape + xdim);
	}

	// Malloc output array and copy values
	malloc_array(y, squeezed_ndims, squeezed_shape);
	copy_array(x, y);
	free(squeezed_shape);
	free(squeezed_dims);
}
void unsqueeze(NDArray* x, NDArray* y, int dim) {
}
void permute(NDArray* x, NDArray* y, int* dims) {
}
void flatten(NDArray* x, NDArray* y) {
}
void reshape(NDArray* x, NDArray* y, int* shape) {
}

// Joining
void stack(NDArray** x_list, NDArray* y, int dim) {
}
void cat(NDArray** x_list, NDArray* y, int dim) {
}


// ---
// Arithmetic operators

// Point-wise unary
void log_(NDArray* x, NDArray* y) {
	for (int idx=0; idx<x->numel; idx++)
		*(y->arr + idx) = logf(*(x->arr + idx));
}

// Point-wise binary
void add(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->numel; idx++)
		*(y->arr + idx) = *(x1->arr + idx) + *(x2->arr + idx);
}
void sub(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->numel; idx++)
		*(y->arr + idx) = *(x1->arr + idx) - *(x2->arr + idx);
}
void mul(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->numel; idx++)
		*(y->arr + idx) = *(x1->arr + idx) * *(x2->arr + idx);
}
void truediv(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->numel; idx++)
		*(y->arr + idx) = *(x1->arr + idx) / *(x2->arr + idx);
}
void pow_(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->numel; idx++)
		*(y->arr + idx) = pow(*(x1->arr + idx), *(x2->arr + idx));
}

// Shape-altering unary / reduction
void sum(NDArray* x, NDArray* y, int* dims, int ndims) {
}
void mean(NDArray* x, NDArray* y, int* dims) {
}
void prod(NDArray* x, NDArray* y, int* dims) {
}

// Shape-altering binary
void vecdot(NDArray* x1, NDArray* x2, NDArray* y) {
	// TODO
}
void matmul(NDArray* x1, NDArray* x2, NDArray* y) {
	// TODO
}
void conv1d(NDArray* x, NDArray* k, NDArray* y) {
	// TODO
}
void conv2d(NDArray* x, NDArray* k, NDArray* y) {
	// TODO
}


#endif /* CPUARRAYS_H */
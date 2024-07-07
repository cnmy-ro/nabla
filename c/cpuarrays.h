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
	int size;  // Num elements in arr
	float* arr;
};
typedef struct NDArray NDArray;


// ---
// Core utils

int calc_size(int ndims, int* shape) {
	int size = 1;
	for (int i=0; i<ndims; i++)
		size = size * (*(shape + i));
	return size;
}
void malloc_array(NDArray* x, int ndims, int* shape) {
	
	// Init metadata variables
	x->ndims = ndims;
	x->shape = malloc(ndims * sizeof(int));	
	for (int i=0; i<ndims; i++)
		*(x->shape + i) = *(shape + i);
	x->size = calc_size(ndims, shape);

	// Dynamically allocate array memory
	x->arr = malloc(x->size * sizeof(float));
}
void free_array(NDArray* x) {
	free(x->arr);
	free(x->shape);
	x->size = 0;
	x->ndims = 0;
}
void copy_array(NDArray* x, NDArray* y) {
	for (int idx=0; idx<x->size; idx++)
		*(y->arr + idx) = *(x->arr + idx);
}
void print_array(NDArray* x) {
	for (int idx=0; idx<x->size; idx++)
		printf("%f ", *(x->arr + idx));
	// TODO: format properly
}


// ---
// Initialization routines

void init_array_full(NDArray* x, float fill_value) {
	for (int idx=0; idx<x->size; idx++)
		*(x->arr + idx) = fill_value;
}
void init_array_linspace(NDArray* x, float start, float end, int steps) {
	for (int idx=0; idx<x->size; idx++)
		*(x->arr + idx) = (1 - idx/steps)*start + (idx/steps)*end;
}
void init_array_rand(NDArray* x) {
	srand(time(NULL));
	for (int idx=0; idx<x->size; idx++)
		*(x->arr + idx) = rand() / (float)RAND_MAX;
}
void init_array_randn(NDArray* x) {
	// TODO
	// Use Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}
void init_array_randint(NDArray* x, float start, float end) {
	srand(time(NULL));
	float randint;
	for (int idx=0; idx<x->size; idx++) {
		randint = (rand() / (float)RAND_MAX) * (end - start) + start;
		*(x->arr + idx) = rand() / (float)RAND_MAX;
	}
}


// ---
// Shape operators

// Indexing / slicing
void slice(NDArray* x, NDArray* y, int dim, int idx) {
}

// Mutation
void squeeze(NDArray* x, NDArray* y) {
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
void stack(NDArray* xs, NDArray* y, int dim) {
}
void cat(NDArray* xs, NDArray* y, int dim) {
}

// Reduction
void sum(NDArray* x, NDArray* y, int* dims) {
}
void mean(NDArray* x, NDArray* y, int* dims) {
}
void prod(NDArray* x, NDArray* y, int* dims) {
}


// ---
// Arithmetic operators

// Point-wise unary
void log_(NDArray* x, NDArray* y) {
	for (int idx=0; idx<x->size; idx++)
		*(y->arr + idx) = logf(*(x->arr + idx));
}

// Point-wise binary
void add(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->size; idx++)
		*(y->arr + idx) = *(x1->arr + idx) + *(x2->arr + idx);
}
void sub(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->size; idx++)
		*(y->arr + idx) = *(x1->arr + idx) - *(x2->arr + idx);
}
void mul(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->size; idx++)
		*(y->arr + idx) = *(x1->arr + idx) * *(x2->arr + idx);
}
void truediv(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->size; idx++)
		*(y->arr + idx) = *(x1->arr + idx) / *(x2->arr + idx);
}
void pow1(NDArray* x, float p, NDArray* y) {
	for (int idx=0; idx<x->size; idx++)
		*(y->arr + idx) = pow(*(x->arr + idx), p);
}
void pow2(NDArray* x1, NDArray* x2, NDArray* y) {
	for (int idx=0; idx<x1->size; idx++)
		*(y->arr + idx) = pow(*(x1->arr + idx), *(x2->arr + idx));
}
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
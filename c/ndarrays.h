#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef NDARRAYS_H
#define NDARRAYS_H


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
	int size = calc_size(ndims, shape);
	x->size = size;

	// Dynamic allocate memory
	x->arr = malloc(size * sizeof(float));
}

void free_array(NDArray* x) {
	free(x->arr);
	free(x->shape);
	x->size = 0;
	x->ndims = 0;
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


// ---
// Shape operators

// Indexing / slicing
void slice(NDArray* x, NDArray* y, int dim, int idx) {
}

// Mutation
void permute(NDArray* x, int* dims) {
}
void flatten(NDArray* x) {
}
void reshape(NDArray* x, int* shape) {
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


#endif /* NDARRAYS_H */
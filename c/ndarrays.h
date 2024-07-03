#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef NDARRAYS_H
#define NDARRAYS_H


// ---
// Core data structure

struct NDArray {	
	size_t ndims;
	size_t* shape;
	float* arr;
};
typedef struct NDArray NDArray;


// ---
// Core utils

size_t calc_total_values(size_t ndims, size_t* shape) {
	size_t num_values = 1;
	for (size_t i=0; i<ndims; i++)
		num_values = num_values * (*(shape + i));
	return num_values;
}

void malloc_array(NDArray* x, size_t ndims, size_t* shape) {

	// Dynamic allocate memory
	size_t num_values = calc_total_values(ndims, shape);
	x->shape = malloc(ndims * sizeof(size_t));	
	x->arr = malloc(num_values * sizeof(float));

	// Init metadata variables
	x->ndims = ndims;
	for (size_t i=0; i<ndims; i++)
		*(x->shape + i) = *(shape + i);
}
void free_array(NDArray* x) {
	free(x->arr);
	free(x->shape);
	x->shape = 0;
}
void print_array(NDArray* x) {
	size_t num_values = calc_total_values(x->ndims, x->shape);
	for (int idx=0; idx<num_values; idx++)
		printf("%f ", *(x->arr + idx));
	// TODO: format properly
}

// ---
// Initialization routines

NDArray fill(size_t ndims, size_t* shape, float fill_value) {
	NDArray x;
	malloc_array(&x, ndims, shape);	
	size_t num_values = calc_total_values(ndims, shape);
	for (int idx=0; idx<num_values; idx++)
		*(x.arr + idx) = fill_value;
	return x;
}


#endif /* NDARRAYS_H */
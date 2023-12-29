#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef ARRAYS_H
#define ARRAYS_H


// ---
// Core data structure

struct Array {	
	float* data;
	int shape[2];
};
typedef struct Array Array;


// ---
// Utils

void alloc_array(Array* x, int nrows, int ncols) {	
	x->data = malloc(nrows * ncols * sizeof(float));
	x->shape[0] = nrows;
	x->shape[1] = ncols;
}
void free_array(Array* x) {
	free(x->data);
	x->shape[0] = 0;
	x->shape[1] = 0;
}

void fill_array_constant(Array* x, float fill_value) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->data + i*ncols + j) = fill_value;
	}
}
void fill_array_uniform_random(Array* x) {
	srand(time(NULL));
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->data + i*ncols + j) = rand() / (float)RAND_MAX;
	}
}
void fill_array_gaussian_random(Array* x) {
	// TODO
	// Use Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}
void copy_array(Array* y, Array* x) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x->data + idx);
		}
	}
}


void print_array(Array* x) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) 
			printf("%f ", *(x->data + i*ncols + j));
		printf("\n");
	}
}


// ---
// Operators

void row_slice(Array* x, Array* y, int row_idx) {
	int ncols = x->shape[1];
	alloc_array(y, ncols, 1);
	for (int j=0; j<ncols; j++)
		*(y->data + j) = *(x->data + row_idx*ncols + j);
}

void col_slice(Array* x, Array* y, int col_idx) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	alloc_array(y, nrows, 1);
	for (int i=0; i<nrows; i++)
		*(y->data + i) = *(x->data + i*ncols + col_idx);
}

void add(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) + *(x2->data + idx);
		}
	}
}

void sub(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) - *(x2->data + idx);
		}
	}
}

void mul(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) * *(x2->data + idx);
		}
	}
}

void truediv(Array* x1, Array* x2, Array* y) {
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) / *(x2->data + idx);
		}
	}
}

void pow_(Array* x, float p, Array* y) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->data + idx) = pow(*(x->data + idx), p);
		}
	}
}

void log_(Array* x, Array* y) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->data + idx) = logf(*(x->data + idx));
		}
	}
}

void sum(Array* x, Array* y) {	
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	alloc_array(y, 1, 1);
	fill_array_constant(y, 0);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->data) += *(x->data + idx);
		}
	}
}

void dot(Array* x1, Array* x2, Array* y) {
	Array prod;
	mul(x1, x2, &prod);
	sum(&prod, y);
	free_array(&prod);
}

void matmul(Array* x1, Array* x2, Array* y) {
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	alloc_array(y, nrows, ncols);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			Array x1_row, x2_col, dot_prod;
			row_slice(x1, &x1_row, i);
			col_slice(x2, &x2_col, j);
			dot(&x1_row, &x2_col, &dot_prod);
			idx = i*ncols + j;
			*(y->data + idx) = *(dot_prod.data);
			free_array(&x1_row);
			free_array(&x2_col);
		}
	}
}

void conv1d(Array* x, Array* ker, Array* y) {
	// TODO
}

#endif /* ARRAYS_H */
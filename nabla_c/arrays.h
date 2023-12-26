#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define ARRAYS_H


// ---
// Core data structure

struct Array {
	int shape[2];
	float* data;
};
typedef struct Array Array;


// ---
// Utils

void init_array_full(Array* x, int nrows, int ncols, float fill_value) {	
	x->shape[0] = nrows;
	x->shape[1] = ncols;
	x->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->data + i*ncols + j) = fill_value;
	}
}

void init_array_uniform_random(Array* x, int nrows, int ncols) {
	srand(time(NULL));
	x->shape[0] = nrows;
	x->shape[1] = ncols;
	x->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->data + i*ncols + j) = rand() / (float)RAND_MAX;
	}
}

void init_array_gaussian_random(Array* x, int nrows, int ncols) {
	// TODO
	// Use Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}

void free_array(Array* x) {
	free(x->data);
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
	y->shape[0] = ncols;
	y->shape[1] = 1;
	y->data = malloc(ncols * sizeof(float));
	for (int j=0; j<ncols; j++)
		*(y->data + j) = *(x->data + row_idx*ncols + j);
}

void col_slice(Array* x, Array* y, int col_idx) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = 1;
	y->data = malloc(nrows * sizeof(float));
	for (int i=0; i<nrows; i++)
		*(y->data + i) = *(x->data + i*ncols + col_idx);
}

void add(Array* x1, Array* x2, Array* y) {	
	int idx;
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = ncols;
	y->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) + *(x2->data + idx);
		}
	}
}

void sub(Array* x1, Array* x2, Array* y) {	
	int idx;
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = ncols;
	y->data = malloc(nrows * ncols * sizeof(float));	
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) - *(x2->data + idx);
		}
	}
}

void mul(Array* x1, Array* x2, Array* y) {	
	int idx;
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = ncols;
	y->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) * *(x2->data + idx);
		}
	}
}

void truediv(Array* x1, Array* x2, Array* y) {
	int idx;
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = ncols;
	y->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->data + idx) = *(x1->data + idx) / *(x2->data + idx);
		}
	}
}

void power(Array* x, float* p, Array* y) {
	// TODO
}

void loge(Array* x, Array* y) {
	// TODO
}

void sum(Array* x, float* y) {
	int idx;
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	*y = 0;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*y += *(x->data + idx);
		}
	}
}

void dot(Array* x1, Array* x2, float* y) {
	Array prod;
	mul(x1, x2, &prod);
	sum(&prod, y);
	free_array(&prod);
}

void matmul(Array* x1, Array* x2, Array* y) {
	int idx;
	int nrows = x1->shape[0];
	int ncols = x2->shape[1];
	y->shape[0] = nrows;
	y->shape[1] = ncols;
	y->data = malloc(nrows * ncols * sizeof(float));
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			Array x1_row, x2_col;
			float dot_prod;
			row_slice(x1, &x1_row, i);
			col_slice(x2, &x2_col, j);
			dot(&x1_row, &x2_col, &dot_prod);
			idx = i*ncols + j;
			*(y->data + idx) = dot_prod;
			free_array(&x1_row);
			free_array(&x2_col);
		}
	}
}

void conv1d(Array* x, Array* ker, Array* y) {
	// TODO
}
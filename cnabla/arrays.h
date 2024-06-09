#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef ARRAYS_H
#define ARRAYS_H


// ---
// Core data structure

struct Array {	
	float* arr;
	int shape[2];
};
typedef struct Array Array;


// ---
// Core utils

void malloc_array(Array* x, int nrows, int ncols) {	
	x->arr = malloc(nrows * ncols * sizeof(float));
	x->shape[0] = nrows;
	x->shape[1] = ncols;
}
void free_array(Array* x) {
	free(x->arr);
	x->shape[0] = 0;
	x->shape[1] = 0;
}
void copy_array(Array* y, Array* x) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->arr + idx) = *(x->arr + idx);
		}
	}
}
void print_array(Array* x) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) 
			printf("%f ", *(x->arr + i*ncols + j));
		printf("\n");
	}
}

// ---
// Initialization routines

void init_array_value(Array* x, float fill_value) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->arr + i*ncols + j) = fill_value;
	}
}
void init_array_linspace(Array* x, float start, float end, int steps) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(x->arr + idx) = (1 - idx/steps)*start + (idx/steps)*end;
		}
	}
}
void init_array_rand(Array* x) {
	srand(time(NULL));
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)
			*(x->arr + i*ncols + j) = rand() / (float)RAND_MAX;
	}
}
void init_array_randn(Array* x) {
	// TODO
	// Use Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}
void init_array_randint(Array* x, int start, int end) {
	srand(time(NULL));
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			randint = (rand() / (float)RAND_MAX) * (end - start) + start;
			*(x->arr + i*ncols + j) = (float)(int)randint;
		}
	}
}


// ---
// Operators

void row_slice(Array* x, Array* y, int row_idx) {
	int ncols = x->shape[1];
	for (int j=0; j<ncols; j++)
		*(y->arr + j) = *(x->arr + row_idx*ncols + j);
}

void col_slice(Array* x, Array* y, int col_idx) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	for (int i=0; i<nrows; i++)
		*(y->arr + i) = *(x->arr + i*ncols + col_idx);
}

void add(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->arr + idx) = *(x1->arr + idx) + *(x2->arr + idx);
		}
	}
}

void sub(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->arr + idx) = *(x1->arr + idx) - *(x2->arr + idx);
		}
	}
}

void mul(Array* x1, Array* x2, Array* y) {	
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++) {
			idx = i*ncols + j;
			*(y->arr + idx) = *(x1->arr + idx) * *(x2->arr + idx);
		}
	}
}

void truediv(Array* x1, Array* x2, Array* y) {
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->arr + idx) = *(x1->arr + idx) / *(x2->arr + idx);
		}
	}
}

void pow_(Array* x, float p, Array* y) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->arr + idx) = pow(*(x->arr + idx), p);
		}
	}
}

void log_(Array* x, Array* y) {
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->arr + idx) = logf(*(x->arr + idx));
		}
	}
}

void sum(Array* x, Array* y) {	
	int nrows = x->shape[0];
	int ncols = x->shape[1];
	init_array_value(y, 0);
	int idx;
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{
			idx = i*ncols + j;
			*(y->arr) += *(x->arr + idx);
		}
	}
}

void dot(Array* x1, Array* x2, Array* y) {
	Array prod;
	malloc_array(&prod, x1->shape[0], x1->shape[1]);
	mul(x1, x2, &prod);
	sum(&prod, y);
	free_array(&prod);
}

void matmul(Array* x1, Array* x2, Array* y) {
	int nrows = x1->shape[0];
	int ncols = x1->shape[1];
	int idx;
	Array x1_row, x2_col, dotprod;
	malloc_array(&x1_row, ncols, 1);
	malloc_array(&x2_col, nrows, 1);
	malloc_array(&dotprod, 1, 1);
	for (int i=0; i<nrows; i++) {
		for (int j=0; j<ncols; j++)	{			
			row_slice(x1, &x1_row, i);
			col_slice(x2, &x2_col, j);
			dot(&x1_row, &x2_col, &dotprod);
			idx = i*ncols + j;
			*(y->arr + idx) = *(dotprod.arr);			
		}
	}
	free_array(&x1_row);
	free_array(&x2_col);
	free_array(&dotprod);
}

void conv1d(Array* x, Array* ker, Array* y) {
	// TODO
}

#endif /* ARRAYS_H */
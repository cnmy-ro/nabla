#include <string.h>

#ifndef ARRAYS_H
#define ARRAYS_H
#include "arrays.h"
#endif


// ---
// Core data structure

struct Tensor {
    Array data;
    Array grad;
    short int requires_grad;
    struct Tensor* parents[2];
    char func_name[10];
};
typedef struct Tensor Tensor;


// ---
// Utils

void init_tensor(Tensor* x, int nrows, int ncols, float fill_value) {
}

void free_tensor(Tensor* x) {
}

void zero_grad(Tensor* x) {
}


// ---
// Operators

void add_fx(Tensor* x1, Tensor* x2, Tensor* fx) {
}
void add_vjp(Tensor* x1, Tensor* x2, Tensor* fx, Array* vjp_x1, Array* vjp_x2) {
}


// ---
// Backward function

void backward(Tensor* x) {
}

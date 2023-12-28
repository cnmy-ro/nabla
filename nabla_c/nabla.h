#include <string.h>
#include <stdbool.h>
#include "arrays.h"

#ifndef NABLA_H
#define NABLA_H


// ---
// Core data structure

struct Tensor {
    Array data;
    Array grad;
    int shape[2];
    bool requires_grad;
    struct Tensor* parents[2];
    char func_name[10];
};
typedef struct Tensor Tensor;


// ---
// Utils

void alloc_tensor(Tensor* x, int nrows, int ncols, bool requires_grad) {
    alloc_array(&(x->data), nrows, ncols);
    alloc_array(&(x->grad), nrows, ncols);
    x->shape[0] = nrows;
    x->shape[1] = ncols;
    x->requires_grad = requires_grad;
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->func_name, "");
}
void free_tensor(Tensor* x) {
    free_array(&(x->data));
    free_array(&(x->grad));
    x->shape[0] = 0;
    x->shape[1] = 0;
    x->requires_grad = false;
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->func_name, "");
}

void fill_tensor_constant(Tensor* x, float fill_value) {
    fill_array_constant(&(x->data), fill_value);
    fill_array_constant(&(x->grad), 0);
}
void fill_tensor_uniform_random(Tensor* x) {
    fill_array_uniform_random(&(x->data));
    fill_array_constant(&(x->grad), 0);
}
void fill_tensor_gaussian_random(Tensor* x) {
    fill_array_gaussian_random(&(x->data));
    fill_array_constant(&(x->grad), 0);
}

void zero_grad(Tensor* x) {
    fill_array_constant(&(x->grad), 0);
}
void detach(Tensor* x) {
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->func_name, "");
}


// ---
// Operators

void mul_fx(Tensor* x1, Tensor* x2, Tensor* fx) {    
    alloc_tensor(fx, x1->shape[0], x1->shape[1], true);
    mul(&(x1->data), &(x2->data), &(fx->data));
    fill_array_constant(&(fx->grad), 0);
    fx->parents[0] = x1;
    fx->parents[1] = x2;
    strcpy(fx->func_name, "mul_fx");
}
void mul_vjp(Tensor* fx, Array* x1_vjp, Array* x2_vjp) {
    alloc_array(x1_vjp, fx->parents[0]->shape[0], fx->parents[0]->shape[1]);
    alloc_array(x2_vjp, fx->parents[1]->shape[0], fx->parents[1]->shape[1]);
    mul(&(fx->grad), &(fx->parents[1]->data), x1_vjp);
    mul(&(fx->grad), &(fx->parents[0]->data), x2_vjp);
}

void sum_fx(Tensor* x, Tensor* fx) {    
    alloc_tensor(fx, 1, 1, true);
    sum(&(x->data), &(fx->data));
    fill_array_constant(&(fx->grad), 0);
    fx->parents[0] = x;
    strcpy(fx->func_name, "sum_fx");
}
void sum_vjp(Tensor* fx, Array* x_vjp) {
    alloc_array(x_vjp, fx->parents[0]->shape[0], fx->parents[0]->shape[1]);
    fill_array_constant(x_vjp, *(fx->grad.data));
}


// ---
// Backward function

void backward(Tensor* x, Array* grad) {
    
    // Accumulate incoming grad (may/may not be VJP)
    add(&(x->grad), grad, &(x->grad));

    // Compute VJP for parents
    Array parents_vjp[2];
    if (strcmp(x->func_name, "mul_fx") == 0)
        mul_vjp(x, &(parents_vjp[0]), &(parents_vjp[1]));
    else if (strcmp(x->func_name, "sum_fx") == 0)
        sum_vjp(x, &(parents_vjp[0]));

    // Recursively call backward() for each required parent
    for (int p=0; p<2; p++) {
        
        if (x->parents[p] == NULL)
            continue;
        if (x->parents[p]->requires_grad == false)
            continue;
        
        backward(x->parents[p], &(parents_vjp[p]));
    }
}


#endif  /* NABLA_H */
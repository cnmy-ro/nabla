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
    char op_name[10];
};
typedef struct Tensor Tensor;


// ---
// Utils

void malloc_tensor(Tensor* x, int nrows, int ncols, bool requires_grad) {
    malloc_array(&(x->data), nrows, ncols);
    malloc_array(&(x->grad), nrows, ncols);
    x->shape[0] = nrows;
    x->shape[1] = ncols;
    x->requires_grad = requires_grad;
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->op_name, "");
}
void free_tensor(Tensor* x) {
    free_array(&(x->data));
    free_array(&(x->grad));
    x->shape[0] = 0;
    x->shape[1] = 0;
    x->requires_grad = false;
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->op_name, "");
}

void init_tensor_full(Tensor* x, float fill_value) {
    init_array_full(&(x->data), fill_value);
    init_array_full(&(x->grad), 0);
}
void init_tensor_rand(Tensor* x) {
    init_array_rand(&(x->data));
    init_array_full(&(x->grad), 0);
}
void init_tensor_randn(Tensor* x) {
    init_array_randn(&(x->data));
    init_array_full(&(x->grad), 0);
}
void init_tensor_randint(Tensor* x, int start, int end) {
    init_array_randint(&(x->data), start, end);
    init_array_full(&(x->grad), 0);
}

void zero_grad(Tensor* x) {
    init_array_full(&(x->grad), 0);
}
void detach(Tensor* x) {
    x->parents[0] = NULL;
    x->parents[1] = NULL;
    strcpy(x->op_name, "");
}

void print_tensor(Tensor* x) {
    print_array(&(x->data));
}

// ---
// Operators

void mul_fx(Tensor* x1, Tensor* x2, Tensor* y) {
    mul(&(x1->data), &(x2->data), &(y->data));
    init_array_full(&(y->grad), 0);
    y->parents[0] = x1;
    y->parents[1] = x2;
    strcpy(y->op_name, "mul_fx");
}
void mul_vjp(Tensor* y, Array* x1_vjp, Array* x2_vjp) {
    mul(&(y->grad), &(y->parents[1]->data), x1_vjp);
    mul(&(y->grad), &(y->parents[0]->data), x2_vjp);
}

void sum_fx(Tensor* x, Tensor* y) {
    sum(&(x->data), &(y->data));
    init_array_full(&(y->grad), 0);
    y->parents[0] = x;
    strcpy(y->op_name, "sum_fx");
}
void sum_vjp(Tensor* y, Array* x_vjp) {    
    init_array_full(x_vjp, *(y->grad.arr));
}


// ---
// Backward function

bool is_leaf_tensor(Tensor* x) {
    if ((x->parents[0] == NULL) && (x->parents[1] == NULL))
        return true;
    else
        return false;
}
bool is_unary_op(char* op_name) {
    if (strcmp(op_name, "sum_fx") == 0)
        return true;
    else
        return false;
}
bool is_binary_op(char* op_name) {
    if ((strcmp(op_name, "add_fx") == 0) || (strcmp(op_name, "mul_fx") == 0))
        return true;
    else
        return false;
}

void backward(Tensor* x, Array* grad) {
    
    // Accumulate incoming grad (may/may not be VJP)
    add(&(x->grad), grad, &(x->grad));

    // Compute VJP for parents
    int num_parents;
    if (is_leaf_tensor(x))
        num_parents = 0;
    else if (is_unary_op(x->op_name))
        num_parents = 1;
    else if (is_binary_op(x->op_name))
        num_parents = 2;
    
    Array parent_vjps[num_parents];
    for (int p=0; p<num_parents; p++)
        malloc_array(&parent_vjps[p], x->parents[p]->shape[0], x->parents[p]->shape[1]);
    if (strcmp(x->op_name, "mul_fx") == 0)
        mul_vjp(x, &(parent_vjps[0]), &(parent_vjps[1]));
    else if (strcmp(x->op_name, "sum_fx") == 0)
        sum_vjp(x, &(parent_vjps[0]));

    // Recursively call backward() for each required parent
    for (int p=0; p<num_parents; p++) {
        
        if (x->parents[p] == NULL)
            continue;
        if (x->parents[p]->requires_grad == false)
            continue;
        
        backward(x->parents[p], &(parent_vjps[p]));
    }
    
    for (int p=0; p<num_parents; p++)
        free_array(&parent_vjps[p]);
}


#endif  /* NABLA_H */
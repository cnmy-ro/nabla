#ifndef NABLA_SCALAR_H
#define NABLA_SCALAR_H

#include <stdio.h>
#include <string.h>

struct Variable
{
	float data;
	float grad;
	short int is_leaf;
	struct Variable* prev[2];
	char func_name[10];
};

struct Variable init_var(float data)
{
	struct Variable x;
	x.data = data;
	x.grad = 0.0;
	x.is_leaf = 1;
	x.prev[0] = NULL;
	x.prev[1] = NULL;
	return x;
}

void backward(struct Variable* y, float grad);

void add_fx(struct Variable* x1, struct Variable* x2, struct Variable* y);
void add_dfx(struct Variable* x1, struct Variable* x2, float* dx);
void mul_fx(struct Variable* x1, struct Variable* x2, struct Variable* y);
void mul_dfx(struct Variable* x1, struct Variable* x2, float* dx);

#endif
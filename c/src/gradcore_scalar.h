#include <stdio.h>
#include <string.h>

struct Var
{
	float data;
	float grad;
	short int is_leaf;
	struct Var* prev[2];
	char func_name[10];
};

struct Var init_var(float data)
{
	struct Var x;
	x.data = data;
	x.grad = 0.0;
	x.is_leaf = 1;
	x.prev[0] = NULL;
	x.prev[1] = NULL;
	return x;
}


void add_fx(struct Var* x1, struct Var* x2, struct Var* y);
void add_dfx(struct Var* x1, struct Var* x2, float* dx);

void mul_fx(struct Var* x1, struct Var* x2, struct Var* y);
void mul_dfx(struct Var* x1, struct Var* x2, float* dx);

void backward(struct Var* y, float grad);
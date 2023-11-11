#include <string.h>


// ---
// Variable and backward()

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

// ---
// Operators

void add_fx(struct Variable* x1, struct Variable* x2, struct Variable* y)
{		
	y->data = x1->data + x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "add");
};
void add_dfx(struct Variable* x1, struct Variable* x2, float* dx)
{	
	*dx = 1.0;
	*(dx + 1) = 1.0;
};

void mul_fx(struct Variable* x1, struct Variable* x2, struct Variable* y)
{		
	y->data = x1->data * x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "mul");
};
void mul_dfx(struct Variable* x1, struct Variable* x2, float* dx)
{	
	*dx = x2->data;
	*(dx + 1) = x1->data;
};

void backward(struct Variable* y, float grad)
{	
	float local_deriv[2], prev_var_grad;

	y->grad += grad;

	if (y->is_leaf != 1)
	{		
		
		if (strcmp(y->func_name, "add") == 0)
        	add_dfx(y->prev[0], y->prev[1], local_deriv);
        if (strcmp(y->func_name, "mul") == 0)
        	mul_dfx(y->prev[0], y->prev[1], local_deriv);

        for (int i=0; i<2; i++)
        {
        	prev_var_grad = y->grad * local_deriv[i];
        	backward(y->prev[i], prev_var_grad);
        }

	}
}
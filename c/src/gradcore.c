#include <math.h>
#include <stdio.h>
#include <string.h>



// ---
// ndarrays

struct Vector
{
	float data[128];
	int shape[1];
};
struct Matrix
{
	float data[128][128];
	int shape[2];
};
void dot()
{
	// TODO
}



// ---
// gradcore var and backward function

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

void backward(struct Var* y, float grad)
{	
	y->grad += grad;

	if (y->is_leaf != 1)
	{	
		if (strcmp(y->func_name, "neg")) int num_prev_vars = 1;
		else int num_prev_vars = 2;
		float local_deriv[num_prev_vars];
		
	
		if (strcmp(y->func_name, "neg") == 0)
        	neg_dfx(y->prev[0], local_deriv);
		else if (strcmp(y->func_name, "add") == 0)
        	add_dfx(y->prev[0], y->prev[1], local_deriv);
        else if (strcmp(y->func_name, "sub") == 0)
        	sub_dfx(y->prev[0], y->prev[1], local_deriv);
        else if (strcmp(y->func_name, "mul") == 0)
        	mul_dfx(y->prev[0], y->prev[1], local_deriv);
        else if (strcmp(y->func_name, "div") == 0)
        	div_dfx(y->prev[0], y->prev[1], local_deriv);

    
        for (int i=0; i<num_prev_vars; i++)
        {
        	float prev_var_grad = y->grad * local_deriv[i];
        	backward(y->prev[i], prev_var_grad);
        }
	}
}



// ---
// gradcore var ops

void neg_fx(struct Var* x, struct Var* y)
{		
	y->data = -x->data;
	y->is_leaf = 0;
	y->prev[0] = x;
	strcpy(y->func_name, "neg");
};
void neg_dfx(struct Var* x1, struct Var* x2, float* dx)
{	
	*dx = -1.0;
};

void add_fx(struct Var* x1, struct Var* x2, struct Var* y)
{		
	y->data = x1->data + x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "add");
};
void add_dfx(struct Var* x1, struct Var* x2, float* dx)
{	
	*dx = 1.0;
	*(dx + 1) = 1.0;
};

void sub_fx(struct Var* x1, struct Var* x2, struct Var* y)
{		
	y->data = x1->data - x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "sub");
};
void sub_dfx(struct Var* x1, struct Var* x2, float* dx)
{	
	*dx = 1.0;
	*(dx + 1) = -1.0;
};

void mul_fx(struct Var* x1, struct Var* x2, struct Var* y)
{		
	y->data = x1->data * x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "mul");
};
void mul_dfx(struct Var* x1, struct Var* x2, float* dx)
{	
	*dx = x2->data;
	*(dx + 1) = x1->data;
};

void div_fx(struct Var* x1, struct Var* x2, struct Var* y)
{		
	y->data = x1->data / x2->data;
	y->is_leaf = 0;
	y->prev[0] = x1;
	y->prev[1] = x2;
	strcpy(y->func_name, "div");
};
void mul_dfx(struct Var* x1, struct Var* x2, float* dx)
{	
	*dx = 1.0 / x2->data;
	*(dx + 1) = -x1->data / (x2->data^2);
};




// ---

int main()
{	
	
	struct Var x1 = init_var(2.0);
	struct Var x2 = init_var(3.0);
	struct Var x3 = init_var(4.0);
	struct Var p = init_var(0.0);
	struct Var y = init_var(0.0);

	mul_fx(&x1, &x2, &p);
	add_fx(&x3, &p, &y);
	backward(&y, 1.0);

	printf("%f  %f\n", x1.data, x2.data);
	printf("%f\n", y.data);
	printf("%f\n", y.grad);
	printf("%f  %f %f  %f\n", x1.grad, x2.grad, p.grad, x3.grad);

	return 0;
}
#include <stdio.h>
#include "nabla_scalar.h"



int main()
{		
	struct Variable x1 = init_var(2.0);
	struct Variable x2 = init_var(3.0);
	struct Variable x3 = init_var(4.0);
	struct Variable p = init_var(0.0);
	struct Variable y = init_var(0.0);

	mul_fx(&x1, &x2, &p);
	add_fx(&x3, &p, &y);
	backward(&y, 1.0);

	printf("%f  %f  %f\n", x1.data, x2.data, x3.data);
	printf("%f\n", y.data);
	printf("%f\n", y.grad);
	printf("%f  %f  %f  %f\n", x1.grad, x2.grad, p.grad, x3.grad);

	return 0;
}
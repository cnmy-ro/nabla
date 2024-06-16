#include <stdio.h>
#include <stdbool.h>
#include "arrays.h"
#include "nabla.h"


void test_arrays(){

	Array x1, x2, y;
	malloc_array(&x1, 3, 1);
	malloc_array(&x2, 3, 1);
	malloc_array(&y, 3, 1);
	
	init_array_value(&x1, 7);
	init_array_value(&x2, 3);
	
	mul(&x1, &x2, &y);
	print_array(&y);
	printf("\n");

	free_array(&x1);
	free_array(&x2);
	free_array(&y);
}

void test_nabla(){

	// Tensor t1, t2;
	// malloc_tensor(&t1, 3, 1, false);
	// malloc_tensor(&t2, 3, 1, false);
	// fill_tensor_constant(&t1, 2);
	// fill_tensor_constant(&t2, 3);
	// print_array(&(t1.data));
	// printf("\n");
	// print_array(&(t2.data));
	// printf("\n");

	// Tensor t3;
	// add_fx(&t1, &t2, &t3);
	// print_array(&(t3.data));

	Tensor x1, x2, x3, y;
	malloc_tensor(&x1, 3, 1, true);
	malloc_tensor(&x2, 3, 1, true);
	malloc_tensor(&x3, 3, 1, true);
	malloc_tensor(&y, 1, 1, true);
	
	init_tensor_value(&x1, 2);	
	init_tensor_value(&x2, 3);
	mul_fx(&x1, &x2, &x3);
	sum_fx(&x3, &y);
	// printf("%s \n", "-");

	// print_array(&(x3.data));
	// printf("\n");
	// print_array(&(y.data));
	// printf("\n");
	// printf("\n");
	
	// printf("---3");
	Array init_grad;
	// printf("---2");
	malloc_array(&init_grad, 1, 1);
	// printf("---1");
	init_array_value(&init_grad, 1);
	// printf("--0");
	backward(&y, &init_grad);
	print_array(&(x3.grad));
	printf("\n");
	print_array(&(x1.grad));
	printf("\n");
	print_array(&(x2.grad));

	free_array(&init_grad);
	free_tensor(&x1);
	free_tensor(&x2);
	free_tensor(&x3);
	free_tensor(&y);
}

void main() {	

	// Test arrays lib
	test_arrays();

	// Test nabla
	test_nabla();
}
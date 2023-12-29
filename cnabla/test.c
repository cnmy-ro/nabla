#include <stdio.h>
#include <stdbool.h>
#include "arrays.h"
#include "nabla.h"


void test_arrays(){

	Array x1, x2, y;
	alloc_array(&x1, 3, 1);
	alloc_array(&x2, 3, 1);
	alloc_array(&y, 3, 1);
	
	fill_array_constant(&x1, 7);
	fill_array_constant(&x2, 3);
	
	mul(&x1, &x2, &y);
	print_array(&y);
	printf("\n");

	free_array(&x1);
	free_array(&x2);
	free_array(&y);
}

void test_nabla(){

	// Tensor t1, t2;
	// alloc_tensor(&t1, 3, 1, false);
	// alloc_tensor(&t2, 3, 1, false);
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
	alloc_tensor(&x1, 3, 1, true);
	fill_tensor_constant(&x1, 2);
	alloc_tensor(&x2, 3, 1, true);
	fill_tensor_constant(&x2, 3);
	mul_fx(&x1, &x2, &x3);
	sum_fx(&x3, &y);

	print_array(&(x3.data));
	printf("\n");
	print_array(&(y.data));
	printf("\n");
	printf("\n");
	
	Array init_grad;
	alloc_array(&init_grad, 1, 1);
	fill_array_constant(&init_grad, 1);
	backward(&y, &init_grad);
	print_array(&(x3.grad));
	printf("\n");
	print_array(&(x1.grad));
	printf("\n");
	print_array(&(x2.grad));

}

void main() {	

	// Test arrays lib
	test_arrays();

	// Test nabla
	test_nabla();
}
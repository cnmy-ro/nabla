#include <stdio.h>
#include <stdbool.h>
#include "ndarrays.h"

void test_ndarrays(){

	int ndims = 2;
	int shape[2] = {3,4};
	NDArray x;
	malloc_array(&x, ndims, shape);
	init_array_full(&x, 1.0);

	print_array(&x);
	free_array(&x);
}

void main() {
	test_ndarrays();
}
#include <stdio.h>
#include <stdbool.h>
#include "ndarrays.h"

void test_ndarrays(){

	size_t ndims = 2;
	size_t shape[2] = {3,4};
	NDArray x = fill(ndims, shape, 1);

	print_array(&x);
	free_array(&x);
}

void main() {
	test_ndarrays();
}
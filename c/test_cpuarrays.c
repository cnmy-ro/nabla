#include <stdio.h>
#include <stdbool.h>
#include "cpuarrays.h"

// void test_slice(){
// 	int ndims = 2;
// 	int xshape[2] = {3,4};	
// 	NDArray x;
// 	malloc_array(&x, 2, xshape);
// 	init_array_linspace(&x, 0, 12, 12);
// 	// print_array(&x);
// 	// printf("%d %d", (x.stride), x.stride[1]);

// 	NDArray y;
// 	int yshape[1] = {4};
// 	malloc_array(&y, 1, yshape);
// 	slice(&x, &y, 0, 0);
// 	printf("%d %zu \n", (y.numel), *(y.stride));
// 	print_array(&y);
	
// 	// // print_array(&x);
// 	// printf("%zu %zu", *(x.stride), *(x.stride+1));

// 	free_array(&x);
// 	free_array(&y);
// }

void test_squeeze(){

	NDArray z, z2;
	int zshape[4] = {3, 1, 4, 1};
	malloc_array(&z, 4, zshape);
	init_array_linspace(&z, 0, 12, 12);
	squeeze(&z, &z2);

	printf("%d ", z.numel);
	printf("%d ", z2.numel);
	printf("%d ", z2.ndims);

	free_array(&z);
	free_array(&z2);
}

void main() {
	// test_slice();
	test_squeeze();
}
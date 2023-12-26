#include <stdio.h>
#include "arrays.h"
#include "nabla.h"


void main() {	

	// Test arrays lib
	// Array mat, vec;
	// init_array_full(&mat, 3, 4, 0);
	// init_array_full(&vec, 4, 1, 1);	
	// printf("%d %d\n", mat.shape[0], mat.shape[1]);
	// printf("%d %d\n", vec.shape[0], vec.shape[1]);

	// Array vec2, vec3;
	// init_array_full(&vec2, 4, 1, 2);
	// add(&vec, &vec2, &vec3);
	// print_array(&vec3);

	// float y;
	// sum(&vec3, &y);
	// printf("%f\n", y);

	// dot(&vec2, &vec3, &y);
	// printf("%f\n", y);

	Array mat1, mat2, mat3;
	init_array_full(&mat1, 3, 2, 1);
	init_array_full(&mat2, 2, 4, 2);
	matmul(&mat1, &mat2, &mat3);
	print_array(&mat1);
	printf("\n");
	print_array(&mat2);
	printf("\n");
	print_array(&mat3);
	printf("\n");

	Array mat;
	init_array_uniform_random(&mat, 3, 2);
	print_array(&mat);

	// Test nabla
}
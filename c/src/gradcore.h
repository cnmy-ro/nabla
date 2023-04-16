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
// TODO
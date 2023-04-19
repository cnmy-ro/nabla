#include <math.h>
#include <stdio.h>
#include <string.h>



// ---
// ndarrays

struct Array1d
{
	float data[128];
	int shape[1];
};
struct Array2d
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
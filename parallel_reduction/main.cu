#include <stdio.h>
#include <stdlib.h>

#include "common.h"


int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");
//
	int size = 1 << 27; //128 Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;
//
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
//
    int a =3;
    int b= 4;
	compare_results(a,b);
    return 0;
}

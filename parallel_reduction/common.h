#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <fstream> 


enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};

//simple initialization
void initialize(int * input, const int array_size,INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

//compare two arrays
void compare_results(int gpu_result, int cpu_result);

//reduction in cpu
int accumulate_cpu(int * input, const int size);

#endif //!COMMON_H


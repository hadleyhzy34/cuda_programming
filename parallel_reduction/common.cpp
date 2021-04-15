#include "common.h"


void initialize(int * input, const int array_size, INIT_PARAM PARAM, int x)
{
    if (PARAM == INIT_ONE)
    {
        for (int i = 0; i < array_size; i++)
        {
            input[i] = 1;
        }
    }
    else if (PARAM == INIT_ONE_TO_TEN){
        for (int i = 0; i < array_size; i++)input[i] = i % 10;
    }
    else if (PARAM == INIT_RANDOM){
        time_t t;
        srand((unsigned)time(&t));
        for (int i = 0; i < array_size; i++)
        {
            input[i] = (int)(rand() & 0xFF);
            //printf("current value of input[%d] is: %d\n", i, input[i]);
        }
    }
}


//cpu sum
int accumulate_cpu(int * input, const int size)
{
    int sum = 0;
    for(int i=0; i<size; i++)
    {
        sum += input[i];
    }
    return sum;
}


//compare results
void compare_results(int gpu_result, int cpu_result)
{
    printf("GPU result : %d , CPU result : %d \n", gpu_result, cpu_result);
    if (gpu_result == cpu_result){
        printf("GPU and CPU results are same \n");
        return;
    }

    printf("GPU and CPU results are different \n");
}




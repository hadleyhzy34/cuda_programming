#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//reduction neighbored pairs kernel
__global__ void redunction_neighbored_pairs(int * input, 
	int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

    //local data block pointer
    int * i_data = input + blockDim.x * blockIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
	{
        int index = 2 * offset * tid;
        /*if(index < blockDim.x){
            //input[index] += input[index + offset];
            i_data[index] += i_data[index + offset];
        }*/
        if(gid*offset*2<size){
            //input[gid*offset*2] += input[gid*offset*2 + offset];
            printf("block id is: %d current offset is: %d current thread id is: %d current index is: %d and input value is: %d and %d\n",blockIdx.x, offset, gid, gid*offset*2, input[gid*offset*2], input[gid*offset*2 + offset]);
            input[gid*offset*2] += input[gid*offset*2 + offset];
        }
        //input[gid*offset*2] += input[gid*offset*2+offset]

		/*if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}*/
        //synchronize all threads inside one block
		__syncthreads();
	}
    
    //for each block, element that is assigned to the first core/thread of block will be the 
    //sum value of this block
	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
        //printf("current block id is: %d, current gid is: %d, temp[%d] = %d\n",blockIdx.x,gid,blockIdx.x,temp[blockIdx.x]);
	}
}

int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");
//
	//int size = 1 << 27; //128 Mb of data
	int size = 256;
    int byte_size = size * sizeof(int);
	int block_size = 128;
//
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
//
    initialize(h_input, size, INIT_RANDOM);
//
//	//get the reduction result from cpu
	int cpu_result = accumulate_cpu(h_input,size);
//
	dim3 block(block_size);
	dim3 grid(size/ block.x);
//
	printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",grid.x, block.x);
//
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);
//
	int * d_input, *d_temp;
//
    cudaMalloc((void**)&d_input,byte_size);
    cudaMalloc((void**)&d_temp, temp_array_byte_size);
//
    cudaMemset(d_temp, 0 , temp_array_byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
//
    redunction_neighbored_pairs <<<grid, block >>>(d_input,d_temp, size);
//
    cudaDeviceSynchronize();
//
    cudaMemcpy(h_ref,d_temp, temp_array_byte_size,cudaMemcpyDeviceToHost);
//
	int gpu_result = 0;
//
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
//
//	//validity check
    compare_results(gpu_result, cpu_result);
//
    cudaFree(d_temp);
    cudaFree(d_input);
//
	free(h_ref);
	free(h_input);
//
    cudaDeviceReset();
	return 0;
}

/*notice this works for any size of array*/
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

	if (gid > size)
		return;

    int offset = blockDim.x * blockIdx.x * 2;
    for(int i = 1; i <= blockDim.x; i *= 2)
    {
        /*2*i the difference between each thread input value index, for instance, first iteration,
          2*i = 2*1 = 2
        first thread add input[0] and input[1] and saves it to input[0],
        second thread adds input[2] and input[3] and saves it to input[2]*/
        int index = 2*i*tid;
        if(2*i*tid < blockDim.x*2){
            /*printf("blockIdx.x: %d offset: %d gid: %d tid: %d index+offset: %d input[index+offset]: %d index+offset+i: %d input[index+offset+i]: %d\n", 
                    blockIdx.x,
                    offset,
                    gid,
                    tid,
                    index+offset,
                    input[index+offset],
                    index+offset+i,
                    input[index+offset+i]);*/
            input[index+offset] += input[index+offset+i];
        }
        __syncthreads();
    }
    
    //for each block, element that is assigned to the first core/thread of block will be the 
    //sum value of this block
	if (tid == 0)
	{
        //printf("final output value is: %d\n",input[gid]);
		temp[blockIdx.x] = input[offset];
        /*if(blockIdx.x == 1){
            printf("current block id and output value is: %d, %d\n", blockIdx.x, temp[blockIdx.x]);
        }*/
        //printf("current block id is: %d, current gid is: %d, temp[%d] = %d\n",blockIdx.x,gid,blockIdx.x,temp[blockIdx.x]);
	}
}

int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");
//
	int size = 1 << 27; //128 Mb of data
	//int size = 1024;
    int byte_size = size * sizeof(int);
	int block_size = 128;
//
	int * cpu_input, *h_ref;
	cpu_input = (int*)malloc(byte_size);
//
    initialize(cpu_input, size, INIT_RANDOM);
//
//	//get the reduction result from cpu
	int cpu_result = accumulate_cpu(cpu_input,size);
//
	dim3 block(block_size);
	dim3 grid((size%(block_size*2)==0)?size/(block_size*2):size/(block_size*2)+1);
//
	printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",grid.x, block.x);
//
    //prepare pointer to collect sum for each block
	int block_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(block_byte_size);
//
	int * gpu_input, *g_ref;
//
    cudaMalloc((void**)&gpu_input,byte_size);
    cudaMalloc((void**)&g_ref, block_byte_size);
//
    cudaMemset(g_ref, 0, block_byte_size);
    cudaMemcpy(gpu_input, cpu_input, byte_size, cudaMemcpyHostToDevice);
//
    redunction_neighbored_pairs <<<grid, block >>>(gpu_input, g_ref, size);
//
    cudaDeviceSynchronize();
//
    cudaMemcpy(h_ref, g_ref, block_byte_size, cudaMemcpyDeviceToHost);
//
	int gpu_result = 0;
//
	for (int i = 0; i < grid.x; i++)
	{
		printf("current index and h_ref value is: %d, %d\n", i, h_ref[i]);
        gpu_result += h_ref[i];
	}
//
//	//validity check
    compare_results(gpu_result, cpu_result);
//
    cudaFree(g_ref);
    cudaFree(gpu_input);
//
	free(h_ref);
	free(cpu_input);
//
    cudaDeviceReset();
	return 0;
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


//handle when there millions of elements for vectors to be processed
__global__ void sum_array_gpu_long(int *a,int *b,int *c,int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //sequentially processing each thread
    while(tid<size){
        c[tid] = a[tid] + b[tid];
        //tid += blockDim.x * gridDim.x;
        if((tid+blockDim.x*gridDim.x)<size){

            printf("max: %d, tid: %d, added_value: %d\n", blockDim.x*gridDim.x, tid, tid+blockDim.x*gridDim.x);
        }
        tid += blockDim.x * gridDim.x;
    }
}


void sum_array_cpu(int *a, int *b, int *c, int size)
{
    for(int i=0;i<size;i++){
        c[i] = a[i] + b[i];
    }
}

bool checkResult(int *a, int *b, int size)
{
    for(int i=0;i<size;i++){
        if(a[i]!=b[i]){
            printf("the %d th current value of a[i] and b[i] is: %d, %d\n",i,a[i],b[i]);
            return false;
        }
        //printf("the current value of a[i] and b[i] are the same\n");
    }
    return true;
}

int main(int argc, char *argv[])
{	
    int size = 100000000;
    printf("size is: %d\n", size);
    int byte_size = size * sizeof(int);

    int *a_input,*b_input,*c_output,*gpu_output;
    a_input = (int*)malloc(byte_size);
    b_input = (int*)malloc(byte_size);
    c_output = (int*)malloc(byte_size);
    gpu_output = (int*)malloc(byte_size);

    for(int i=0;i<size;i++)
    {
        a_input[i] = i;
        b_input[i] = i*2;
    }
    
    //cpu matrix sum calculation
    sum_array_cpu(a_input,b_input,c_output,size);


    int * a_gpu_input, * b_gpu_input, *c_gpu_output;
    cudaMalloc((void**)&a_gpu_input, byte_size);
    cudaMalloc((void**)&b_gpu_input, byte_size);
    cudaMalloc((void**)&c_gpu_output, byte_size);

    cudaMemcpy(a_gpu_input,a_input,byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu_input,b_input,byte_size,cudaMemcpyHostToDevice);

    //dim3 block(block_x,block_y);
    //dim3 grid(dim_x,dim_y);
    
    int grid_size = 65535;
    dim3 block(128);
    dim3 grid(grid_size);

    printf("dimension of each block is: %d, %d\n", block.x, block.y);
    printf("dimension of grid is: %d, %d\n", grid.x, grid.y);
    
    sum_array_gpu_long<<<grid,block>>>(a_gpu_input,b_gpu_input,c_gpu_output,size);
    cudaDeviceSynchronize();
    
    //memory transfer back to host
    cudaMemcpy(gpu_output,c_gpu_output,byte_size,cudaMemcpyDeviceToHost);

    bool test = checkResult(c_output,gpu_output,size);
    if(test==true){
        printf("the result is true\n");
    }else{
        printf("the result is false\n");
    }

    cudaFree(a_gpu_input);
    cudaFree(b_gpu_input);
    cudaFree(c_gpu_output);
    
    free(a_input);
    free(b_input);
	free(c_output);
    cudaDeviceReset();
	
    return 0;
}

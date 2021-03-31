#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

//implement one grid with 4 blocks and 256 threads in total, 8x8 threads for each block
__global__ void print_threadIds()
{
	printf("blockIdx,x : %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y : %d, blockDim.z : %d gridDim.x : %d, gridDim.y : %d, gridDim.z : %d \n",blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}


__global__ void unique_idx_calc_threadIdx(int * input)
{
	int tid = threadIdx.x;
    int offset = (blockIdx.x>0)? 4:0;
	printf("blockIdx : %d, threadIdx : %d, value : %d\n", blockIdx.x, tid, input[tid+offset]);
}


__global__ void unique_gid_calculation(int * input){
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int offset = blockIdx.y * gridDim.x * (blockDim.x * blockDim.y) + blockIdx.x * (blockDim.x * blockDim.y);
    //number of threads in one row = gridDim.x * blockDim.x
    //row offset: gridDim.x * blockDim.x * blockIdx.y
    //int offset = blockIdx.x * (blockDim.x * blockDim.y) + blockIdx.y * (blockDim.x * blockDim.y);
    int gid = tid + offset;
    printf("gid: %d, input[gid]: %d \n",gid, input[gid]);
    printf("threadIdx.x : %d, blockIdx.x : %d, blockIdx.y : %d, blockDim.x : %d, blockDim.y : %d, gridDim.x : %d gid : %d value : %d\n", 
           threadIdx.x, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gid, input[gid]);
}


__global__ void mem_trs_test(int * input)
{
    int n_per_block = blockDim.x * blockDim.y * blockDim.z;
    int offset_grid = blockIdx.z * (gridDim.x*gridDim.y) * n_per_block  + blockIdx.y * gridDim.x * n_per_block + blockIdx.x * n_per_block;
    int offset_block = threadIdx.z * (blockDim.x*blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int gid = offset_grid + offset_block;
    printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}

__global__ void mem_trs_test1(int * input,int size)
{
    int gid = blockIdx.y * (blockDim.x*blockDim.y)*gridDim.x + blockIdx.x * (blockDim.x*blockDim.y) + threadIdx.x;
    //if(gid<size){
    printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
    //}
}




int main()
{	
    int size = 64;
    int byte_size = size * sizeof(int);

    int *h_input;
    h_input = (int*)malloc(byte_size);
    
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<size;i++)
    {
        h_input[i] = (int)(rand() &0xff);
    }
    
    int * d_input;
    cudaMalloc((void**)&d_input, byte_size);

    cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);
    dim3 block(2,2,2);
    dim3 grid(2,2,2);
    mem_trs_test<<<grid,block>>>(d_input);
    //data transfer between host and device
    //direction: 
    //host to device - cudamemcpyhtod
    //device to host - cudamemcpydtoh
    //device to device - cudamemcpydtod

    cudaDeviceSynchronize();
    
    cudaFree(d_input);
    free(h_input);
	
    cudaDeviceReset();
	return 0;
}

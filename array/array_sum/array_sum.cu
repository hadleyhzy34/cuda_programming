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
    int gid = blockIdx.y * (blockDim.x*blockDim.y)*gridDim.x + blockIdx.x * (blockDim.x*blockDim.y) + threadIdx.x;
    printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}

__global__ void mem_trs_test1(int * input,int size)
{
    int gid = blockIdx.y * (blockDim.x*blockDim.y)*gridDim.x + blockIdx.x * (blockDim.x*blockDim.y) + threadIdx.x;
    //if(gid<size){
    printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
    //}
}


__global__ void sum_array_gpu(int *a,int *b,int *c,int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
    printf("gid : %d, a[gid] : %d, b[gid] : %d, c[gid] : %d\n", gid, a[gid], b[gid], c[gid]);
}


int main()
{	
    int size = 10000;
    int block_size = 128;
    int byte_size = size * sizeof(int);

    int *a_input,*b_input,*c_output;
    a_input = (int*)malloc(byte_size);
    b_input = (int*)malloc(byte_size);
    c_output = (int*)malloc(byte_size);

    for(int i=0;i<size;i++)
    {
        a_input[i] = i;
        b_input[i] = i*2;
    }
    
    int * a_gpu_input, * b_gpu_input, *c_gpu_output;
    cudaMalloc((void**)&a_gpu_input, byte_size);
    cudaMalloc((void**)&b_gpu_input, byte_size);
    cudaMalloc((void**)&c_gpu_output, byte_size);

    cudaMemcpy(a_gpu_input,a_input,byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu_input,b_input,byte_size,cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(8);
    sum_array_gpu<<<grid,block>>>(a_gpu_input,b_gpu_input,c_gpu_output,size);

    cudaDeviceSynchronize();
    
    cudaFree(a_gpu_input);
    cudaFree(b_gpu_input);
    // cudaFree(c_gpu_ouput);
    free(a_input);
    free(b_input);
	
    cudaDeviceReset();
	return 0;
}

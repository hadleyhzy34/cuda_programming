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


int main()
{	
	//define number of threads for each dimension
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int cpu_data[] = {23,9,4,53,65,12,1,33,34,51,3,100,2,22,15,99,98};
    
    //printout data from traditional cpu memory
    for(int i=0;i<array_size;i++){
        printf("the %d th element is: %d\n", i, cpu_data[i]);
    }
    printf("\n\n");
    
    //gpu data copied from cpu memory
    int *gpu_data;
    cudaMalloc((void**)&gpu_data, array_byte_size);
    cudaMemcpy(gpu_data, cpu_data, array_byte_size, cudaMemcpyHostToDevice);
    
    //2x2 thread blocks, each has 4 threads
	dim3 block(2,2);
	dim3 grid(2,2);
    
    //printout thread id and each element from one array by using gpu
	//unique_idx_calc_threadIdx <<< grid, block >>> (gpu_data);
	unique_gid_calculation <<<grid,block>>>(gpu_data);
    cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}

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
	printf("threadIdx : %d, value : %d\n", tid, input[tid]);
}

int main()
{	
	//define number of threads for each dimension
    int array_size = 8;
    int array_byte_size = sizeof(int) * array_size;
    int cpu_data[] = {23,9,4,53,65,12,1,33};
    
    //printout data from traditional cpu memory
    for(int i=0;i<array_size;i++){
        printf("the %d th element is: %d\n", i, cpu_data[i]);
    }
    printf("\n\n");
    
    //gpu data copied from cpu memory
    int *gpu_data;
    cudaMalloc((void**)&gpu_data, array_byte_size);
    cudaMemcpy(gpu_data, cpu_data, array_byte_size, cudaMemcpyHostToDevice);
    
    //one thread block which has 8 threads
	dim3 block(8);
	dim3 grid(1);
    
    //printout thread id and each element from one array by using gpu
	unique_idx_calc_threadIdx <<< grid, block >>> (gpu_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}

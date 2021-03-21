//this is a sample CUDA program
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void hello_cuda()
{
	printf("hello cuda world \n");
}


int main(){
	//kernel_name <<<number_of_blocks, thread_per_block>>>(arguments)
	//hello_cuda <<<1,4>>>();
	//...<<<grid,block>>>(argument)
	dim3 block(4);
	dim3 grid(8);
	//hello_cuda <<< grid,block >>>();

	//dynamically allocation number of blocks and threads
	//declare number of blocks,threads for grids and blocks
	//int nx,ny;
	//nx = 16, ny = 4;
	//dim3 grid_dy(16,4);
	//dim3 block_dy(nx/16,ny/4);
	//hello_cuda <<<grid_dy,block_dy>>>();
	//limitations of threads for each block
	//dim3 block_limit(1025,0,0);
	//hello_cuda <<<grid,block_limit>>> ();
	


	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
	//int *a;
	//cudaMalloc(&a,10);
	//cudaFree(a);
        //printf("this is a test");
	//return 0;
}

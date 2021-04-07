#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void print_details_of_warps()
{
    int gid = blockIdx.x * gridDim.x * blockDim.x
        + blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / 32;

    int grid_idx = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid : %d, bid.x : %d, bid.y : %d, gid : %d, warp_id : %d, grid_idx : %d\n",
            threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, grid_idx);
}


__global__ void code_without_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a,b;
    a = b = 0;

    int warp_id = gid / 32;

    if(warp_id % 2 == 0)
    {
        a = 100.0;
        b = 50.0;
    }
    else{
        a = 200;
        b = 75;
    }
}

__global__ void divergence_code()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0;

    if(gid%2 == 0){
        a = 100.0;
        b = 50.0;
    }else{
        a = 200;
        b = 75;
    }
}

int main(int argc, char** argv){
    int size = 1 << 22;

    dim3 block_size(128);
    dim3 grid_size((size+block_size.x-1)/block_size.x);

    code_without_divergence <<< grid_size, block_size>>>();
    cudaDeviceSynchronize();

    divergence_code <<< grid_size, block_size>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>


//#define gpuErrchk(ans) { gpuAssert((ans),__FILE__,__LINE__);}


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
    int size = 1000;
    int block_size = argc;
    int byte_size = size * sizeof(int);
    cudaError error;


    int *a_input,*b_input;
    a_input = (int*)malloc(byte_size);
    b_input = (int*)malloc(byte_size);
    
    int *c_output,*gpu_output;
    c_output = (int*)malloc(byte_size);
    gpu_output = (int*)malloc(byte_size);


    for(int i=0;i<size;i++)
    {
        a_input[i] = i;
        b_input[i] = i*2;
    }
    clock_t cpu_start,cpu_end;
    cpu_start = clock();
    //cpu matrix sum calculation
    sum_array_cpu(a_input,b_input,c_output,size);
    cpu_end = clock();


    int * a_gpu_input, * b_gpu_input, *c_gpu_output;
    error = cudaMalloc((void**)&a_gpu_input, byte_size);
    if(error != cudaSuccess)
    {
        fprintf(stderr,"%s \n", cudaGetErrorString(error));
    }

    cudaMalloc((void**)&b_gpu_input, byte_size);
    cudaMalloc((void**)&c_gpu_output, byte_size);

    clock_t h2d_start,h2d_end;
    h2d_start = clock();
    cudaMemcpy(a_gpu_input,a_input,byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu_input,b_input,byte_size,cudaMemcpyHostToDevice);
    h2d_end = clock();

    dim3 block(block_size);
    dim3 grid(size/block.x+((size%block.x==0)?0:1));

    printf("size of block and grid is: %d, %d\n",block.x,grid.x);
    
    clock_t gpu_start,gpu_end;
    gpu_start = clock();
    sum_array_gpu<<<grid,block>>>(a_gpu_input,b_gpu_input,c_gpu_output,size);
    cudaDeviceSynchronize();
    gpu_end = clock();
    
    clock_t d2h_start,d2h_end;
    d2h_start = clock();
    //memory transfer back to host
    cudaMemcpy(gpu_output,c_gpu_output,byte_size,cudaMemcpyDeviceToHost);
    d2h_end = clock();

    //for(int i=0;i<size;i++){
    //    printf("the gpu_output[i] value is: %d",gpu_output[i]);
    //}

    bool test = checkResult(c_output,gpu_output,size);
    if(test==true){
        printf("the result is true\n");
    }else{
        printf("the result is false\n");
    }
//    if(checkResult(c_gpu_output,c_output,size)==true){
//        printf("the result is correct");
//    }else{
//        printf("the result is not correct");
//    }

    cudaDeviceSynchronize();
    printf("sum array cpu execution of time : %4.6f \n",(double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));
    printf("sum array gpu execution of time : %4.6f \n",(double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
    printf("total sum array gpu execution of time : %4.6f \n",(double)((double)(gpu_end - gpu_start + h2d_end - h2d_start + d2h_end - d2h_start)/CLOCKS_PER_SEC));
    cudaFree(a_gpu_input);
    cudaFree(b_gpu_input);
    cudaFree(c_gpu_output);
    free(a_input);
    free(b_input);
	free(c_output);
    cudaDeviceReset();
	return 0;
}

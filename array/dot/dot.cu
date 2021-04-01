#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define imin(a,b) (a<b?a:b)

//number of elements to be processed
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock );


//handle when there millions of elements for vectors to be processed
__global__ void sum_array_gpu_long(int *a,int *b,int *c,int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid<size){
        c[tid] = a[tid] + b[tid];
        //tid += blockDim.x * gridDim.x;
        if((tid+blockDim.x*gridDim.x)<size){

            printf("max: %d, tid: %d, added_value: %d\n", blockDim.x*gridDim.x, tid, tid+blockDim.x*gridDim.x);
        }
        tid += blockDim.x * gridDim.x;
    }
}


__global__ void dot(float *a,float *b, float *c){
    //shared memory between threads in one block, compiler will create a copy of the shared variables for each block
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while(tid<N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    //set the cache values
    cache[cacheIndex] = temp;

    //synchronize threads in this block
    //this call guarantees that every thread in the block has completed instructions prior
    //to the __syncthreads()
    __syncthreads();

    //parallel to sum all elements in one array
    //for reductions, threadsPerBlock must be a power of 2
    int i = blockDim.x/2;
    while(i != 0){
        if(cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}



int main(int argc, char *argv[])
{	
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    // allocate memory on the CPU side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,N*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b,N*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c,blocksPerGrid*sizeof(float) ) );
    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    // copy the arrays ‘a’ and ‘b’ to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float),cudaMemcpyHostToDevice ) );
    
    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,dev_partial_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost ) );
    // finish up on the CPU side
    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }
    
    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c,2 * sum_squares( (float)(N - 1) ) );
    // free memory on the GPU side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_partial_c );
    // free memory on the CPU side
    free( a );
    free( b );
    free( partial_c );
	
    return 0;
}

#include "dot_product.h"

#define THREADSPERBLOCK 256
__global__ void dotProductKernel(float *a, float *b, float *c, int size,
                                 const int threadsPerBlock) {
  __shared__ float cache[THREADSPERBLOCK];
  // __shared__ float cache[threadsPerBlock];
  // get current threadid
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  float tmp = 0.0;
  while (tid < size) {
    tmp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  printf("block: %d, threadIdx: %d, val: %f\n", blockIdx.x, threadIdx.x, tmp);
  cache[threadIdx.x] = tmp;
  // synchronize threads in this block
  __syncthreads();

  int i = THREADSPERBLOCK / 2;
  while (i > 0) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    i /= 2;
    // printf("%d", i);
    __syncthreads();
  }
  printf("block: %d, threadIdx: %d, c: %f\n", blockIdx.x, threadIdx.x,
         cache[0]);
  if (threadIdx.x == 0) {
    *c += cache[0];
  }
}

void computation::dotProduct(float *a, float *b, float *c, int size) {
  const int bytes = size * sizeof(float);

  // Device arrays
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, sizeof(float));

  // Copy data to device
  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, sizeof(float));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size,
                                                       THREADSPERBLOCK);

  // Copy result back to host
  cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

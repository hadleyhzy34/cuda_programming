#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int N = 1 << 20; // 1 million elements
const int THREADS_PER_BLOCK = 256;

__global__ void dotProductKernel(const float *A, const float *B,
                                 float *partialSums, int N) {
  __shared__ float sharedMem[THREADS_PER_BLOCK]; // Shared memory for reduction

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;

  // Element-wise multiplication
  while (tid < N) {
    sum += A[tid] * B[tid];
    tid += stride;
  }

  // Store thread's partial sum in shared memory
  sharedMem[threadIdx.x] = sum;
  __syncthreads();

  // Block-level reduction using shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Write block's final partial sum to global memory
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = sharedMem[0];
  }
}

int main() {
  // Host memory allocation
  std::vector<float> A(N), B(N);
  for (int i = 0; i < N; ++i) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  // Device memory allocation
  float *d_A, *d_B, *d_partialSums;
  cudaMalloc(&d_A, N * sizeof(float));
  cudaMalloc(&d_B, N * sizeof(float));
  cudaMalloc(&d_partialSums, (N / THREADS_PER_BLOCK) * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dotProductKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_partialSums, N);

  // Reduce partial sums on host
  std::vector<float> partialSums(blocks);
  cudaMemcpy(partialSums.data(), d_partialSums, blocks * sizeof(float),
             cudaMemcpyDeviceToHost);

  float total = 0.0f;
  for (float val : partialSums) {
    total += val;
  }

  std::cout << "Dot Product: " << total << std::endl;

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_partialSums);
  return 0;
}

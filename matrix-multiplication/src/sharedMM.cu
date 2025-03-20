#include "sharedMM.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

const int TILE_SIZE = 16;

__global__ void matrixMulShared(float *C, float *A, float *B, int M, int N,
                                int K) {
  __shared__ float s_A[TILE_SIZE][TILE_SIZE];
  __shared__ float s_B[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  // Loop over tiles of A and B
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tiles into shared memory
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
      s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    else
      s_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
      s_B[threadIdx.y][threadIdx.x] =
          B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
      s_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads(); // Wait for all threads to load the tile

    // Compute partial dot product
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    }

    __syncthreads(); // Ensure tile is fully used before next iteration
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void Computation::runShared(float *h_A, float *h_B, float *h_C, int M, int N,
                            int K) {
  float *d_A, *d_B, *d_C;
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_A, size_A));
  CUDA_CHECK(cudaMalloc(&d_B, size_B));
  CUDA_CHECK(cudaMalloc(&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  dim3 block(TILE_SIZE, TILE_SIZE); // 16x16 = 256 threads
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  auto start = std::chrono::high_resolution_clock::now();
  matrixMulShared<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Shared memory time: " << elapsed.count() << " ms\n";

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

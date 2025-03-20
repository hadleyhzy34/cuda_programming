#include "transpose.hpp"
#include <ctime>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16
// Tile size (must be a power of 2 for simplicity, e.g., 32x32 fits well in
// shared memory)
const int TILE_DIM = 32;
// Padding to avoid bank conflicts in shared memory
const int BLOCK_ROWS = 8;

// __global__ void matrixTransposeShared(float *output, const float *input,
//                                       int width, int height) {
//   // Shared memory tile (with padding to avoid bank conflicts)
//   __shared__ float tile[TILE_DIM][TILE_DIM + 1];
//
//   // Global input coordinates (read from input matrix)
//   int x = blockIdx.x * TILE_DIM + threadIdx.x;
//   int y = blockIdx.y * TILE_DIM + threadIdx.y;
//
//   // Read input into shared memory (coalesced global reads)
//   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
//     if (x < width && (y + j) < height) {
//       tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
//     }
//   }
//
//   __syncthreads(); // Wait for all threads to load the tile
//
//   // Global output coordinates (write to transposed matrix)
//   x = blockIdx.y * TILE_DIM + threadIdx.x; // Swap blockIdx.x and blockIdx.y
//   y = blockIdx.x * TILE_DIM + threadIdx.y;
//
//   // Write from shared memory to output (coalesced global writes)
//   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
//     if (x < height && (y + j) < width) {
//       output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
//       printf("x: %d, y:%d, output: %f\n", x, y, output[(y + j) * height +
//       x]);
//     }
//   }
// }
__global__ void matrixTransposeGlobal(float *d_in, float *d_out, int width,
                                      int height) {
  // Calculate row and column indices for the output matrix
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (row < width && col < height) {
    d_out[col * width + row] = d_in[row * width + col];
  }
}

__global__ void matrixTransposeShared(float *d_in, float *d_out, int width,
                                      int height) {
  // Calculate row and column indices for the output matrix
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // Shared memory for a block of the input matrix
  __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
  // __shared__ float test[blockDim.x][blockDim.y];

  // // Load data from global memory to shared memory
  // int x = col;
  // int y = row;

  if (col < width && row < height) {
    tile[threadIdx.y][threadIdx.x] = d_in[row * width + col];
    // printf("row: %d, col: %d, input: %f\n", x, y, d_in[y * width + x]);
  } else {
    tile[threadIdx.y][threadIdx.x] = 0.0f; // Pad if out of bounds
  }

  // // Transpose the data within shared memory
  // float temp = tile[threadIdx.y][threadIdx.x];
  // tile[threadIdx.y][threadIdx.x] = tile[threadIdx.x][threadIdx.y];
  // tile[threadIdx.x][threadIdx.y] = temp;

  // // Transpose the data within shared memory
  // float temp = tile[threadIdx.y][threadIdx.x];
  // tile[threadIdx.y][threadIdx.x] = tile[threadIdx.x][threadIdx.y];
  // tile[threadIdx.x][threadIdx.y] = temp;

  // Synchronize again to ensure the transpose is complete
  __syncthreads();

  // Write the transposed data from shared memory to global memory
  if (col < width && row < height) {
    d_out[col * width + row] = tile[threadIdx.y][threadIdx.x];
    // printf("transposed, row: %d, col: %d, input: %f\n", x, y,
    //        d_out[col * width + row]);
  }
}

void computation::matrixTranspose(float *h_in, float *h_out, int width,
                                  int height) {
  size_t matrixSize = width * height * sizeof(float);

  float *d_in, *d_out;
  cudaMalloc((void **)&d_in, matrixSize);
  cudaMalloc((void **)&d_out, matrixSize);

  cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch the kernel
  matrixTransposeShared<<<gridDim, blockDim>>>(d_in, d_out, width, height);

  cudaMemcpy(h_out, d_out, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void computation::matrixGlobalTranspose(float *h_in, float *h_out, int width,
                                        int height) {
  size_t matrixSize = width * height * sizeof(float);

  float *d_in, *d_out;
  cudaMalloc((void **)&d_in, matrixSize);
  cudaMalloc((void **)&d_out, matrixSize);

  cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch the kernel
  matrixTransposeGlobal<<<gridDim, blockDim>>>(d_in, d_out, width, height);

  cudaMemcpy(h_out, d_out, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void computation::tranposeComparison(float *h_in, float *h_out_shared,
                                     float *h_out_global, int width,
                                     int height) {
  size_t matrixSize = width * height * sizeof(float);

  float *d_in, *d_out_shared, *d_out_global;
  cudaMalloc((void **)&d_in, matrixSize);
  cudaMalloc((void **)&d_out_global, matrixSize);
  cudaMalloc((void **)&d_out_shared, matrixSize);

  cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // start running time
  clock_t start = clock();

  for (auto i = 0; i < 1000; i++) {
    // Launch the kernel
    matrixTransposeShared<<<gridDim, blockDim>>>(d_in, d_out_shared, width,
                                                 height);
  }
  // // Launch the kernel
  // matrixTransposeShared<<<gridDim, blockDim>>>(d_in, d_out_shared, width,
  //                                              height);

  printf("Time taken for current approach: %.2fs\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);

  start = clock();

  for (auto i = 0; i < 1000; i++) {
    // Launch the kernel
    matrixTransposeGlobal<<<gridDim, blockDim>>>(d_in, d_out_global, width,
                                                 height);
  }
  // matrixTransposeGlobal<<<gridDim, blockDim>>>(d_in, d_out_global, width,
  //                                              height);

  printf("Time taken for current approach: %.2fs\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);

  cudaMemcpy(h_out_shared, d_out_shared, matrixSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_global, d_out_shared, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out_shared);
  cudaFree(d_out_global);
}

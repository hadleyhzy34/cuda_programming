#include "textureSobelFilter.hpp"
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Sobel filter in constant memory
__constant__ float sobelFilter[9];

// Sobel kernel for edge detection
__global__ void textureSobelKernel(cudaTextureObject_t texObj, float *output,
                                   int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  float sum = 0.0f;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      float pixel = tex2D<float>(texObj, col + j, row + i);

      sum += pixel * sobelFilter[(i + 1) * 3 + (j + 1)];
    }
  }
  output[row * width + col] = sum;
}

// Template function to launch any CUDA kernel
void Texture::launchKernel(float h_filter[], float *h_input, float *h_output,
                           int width, int height) {
  // Step 1: Allocate CUDA array
  cudaArray_t cuArray;
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<float>(); // Simplified for single float
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

  // Step 2: Copy host data to CUDA array
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_input, width * sizeof(float),
                                 width * sizeof(float), height,
                                 cudaMemcpyHostToDevice));

  // Step 3: Create and configure texture object
  cudaResourceDesc resDesc{};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc{};
  texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp for x
  texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp for y
  texDesc.filterMode = cudaFilterModePoint;      // No interpolation
  texDesc.readMode = cudaReadModeElementType;    // Raw float values
  texDesc.normalizedCoords = 0;                  // Use absolute coordinates

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

  // Step 4: Allocate output device memory
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(float)));

  // Copy Sobel filter to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(sobelFilter, h_filter, sizeof(float) * 9));

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  textureSobelKernel<<<gridSize, blockSize>>>(texObj, d_output, width, height);

  // Check for any errors during kernel execution
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_input, d_output, sizeof(float) * width * height,
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaDestroyTextureObject(texObj));
  CUDA_CHECK(cudaFreeArray(cuArray));
}

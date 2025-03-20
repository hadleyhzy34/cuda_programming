#include "sobelFilter.hpp"

// Sobel kernel for edge detection
__global__ void SobelKernel(const float *input, float *output, float *filter,
                            int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  float sum = 0.0f;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int imageRow = row + i;
      int imageCol = col + j;
      if (imageRow >= 0 && imageRow < height && imageCol >= 0 &&
          imageCol < width) {
        float pixel = input[imageRow * width + imageCol];
        int filterIdx = (i + 1) * 3 + (j + 1); // 2D to 1D index
        sum += pixel * filter[filterIdx];
      }
    }
  }
  output[row * width + col] = sum;
}

// Template function to launch any CUDA kernel
void SobelFilter::launchKernel(float h_filter[], float *h_input,
                               float *h_output, int width, int height) {

  // Allocate GPU memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, sizeof(float) * width * height);
  cudaMalloc(&d_output, sizeof(float) * width * height);

  // Copy image data to GPU
  cudaMemcpy(d_input, h_input, sizeof(float) * width * height,
             cudaMemcpyHostToDevice);

  // allocate sobel filter to global memory
  float *sobelFilter;
  cudaMalloc(&sobelFilter, sizeof(float) * 9);

  // copy sobel filter from host code to devicde code
  cudaMemcpy(sobelFilter, h_filter, sizeof(float) * 9, cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  SobelKernel<<<gridSize, blockSize>>>(d_input, d_output, sobelFilter, width,
                                       height);

  // Copy result back to host
  cudaMemcpy(h_output, d_output, sizeof(float) * width * height,
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

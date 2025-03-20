#include <cuda_runtime.h>

// Function to measure runtime
double measureRuntime(cudaFunction_t kernel, void *d_in, void *d_out, int width,
                      int height, int blockSize) {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  // Define grid and block dimensions
  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim((width + blockSize - 1) / blockSize,
               (height + blockSize - 1) / blockSize);

  // Launch the kernel
  kernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsedTime;
}

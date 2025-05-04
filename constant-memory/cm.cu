#include <cuda_runtime.h>
#include <iostream>

void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " failed: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Declare constant memory
__constant__ float scale_factor;

__global__ void constantKernel(float *out, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
    out[i] *= scale_factor;
}

__global__ void globalKernel(float *out, float *in, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
    out[i] *= (*in); // Read from global memory
}

int main() {
  const int N = 1 << 20; // 1 million elements
  size_t size = N * sizeof(float);

  // Events for timing
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Host arrays
  float *h_data = new float[N];
  for (int i = 0; i < N; ++i)
    h_data[i] = 1.0f;

  // Device arrays
  float *d_data;
  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

  // Copy scale_factor to constant memory
  float h_scale = 2.0f;
  cudaError_t err = cudaMemcpyToSymbol(scale_factor, &h_scale, sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "failed to copy to constant memory: "
              << cudaGetErrorString(err) << std::endl;
  }

  float *d_in;
  // *h_in = 2.0f;
  cudaMalloc(&d_in, sizeof(float));
  cudaMemcpy(d_in, &h_scale, sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int blockSize = 1024;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Warm-up
  constantKernel<<<1, dim3(32, 32)>>>(d_data, N);
  cudaDeviceSynchronize();

  // Measure naive kernel
  float ms_naive = 0.0f;
  for (int i = 0; i < 100; ++i) {
    cudaEventRecord(start);
    globalKernel<<<gridSize, blockSize>>>(d_data, d_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms_naive += ms;
  }

  // Measure padded kernel
  float ms_padded = 0.0f;
  for (int i = 0; i < 100; ++i) {
    cudaEventRecord(start);
    constantKernel<<<gridSize, blockSize>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms_padded += ms;
  }

  std::cout << "global kernel time: " << ms_naive << " ms" << std::endl;
  std::cout << "constant kernel time: " << ms_padded << " ms" << std::endl;

  // Copy result back to host
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_data);
  delete[] h_data;

  return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel to perform vector addition on mapped host memory
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  // --- 1. Initialization and Device Properties ---
  int n = 1024;
  size_t size = n * sizeof(float);

  // Enable host memory mapping
  checkCudaError(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");

  int deviceId;
  checkCudaError(cudaGetDevice(&deviceId), "cudaGetDevice");

  cudaDeviceProp devProp;
  checkCudaError(cudaGetDeviceProperties(&devProp, deviceId),
                 "cudaGetDeviceProperties");

  if (!devProp.canMapHostMemory) {
    std::cerr << "Device does not support mapping host memory." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Device supports mapping host memory." << std::endl;

  // --- 2. Allocate Mapped Pinned Host Memory ---
  float *h_a, *h_b, *h_c;
  checkCudaError(cudaHostAlloc((void **)&h_a, size, cudaHostAllocMapped),
                 "cudaHostAlloc for h_a");
  checkCudaError(cudaHostAlloc((void **)&h_b, size, cudaHostAllocMapped),
                 "cudaHostAlloc for h_b");
  checkCudaError(cudaHostAlloc((void **)&h_c, size, cudaHostAllocMapped),
                 "cudaHostAlloc for h_c");

  std::cout << "Mapped pinned host memory allocated." << std::endl;

  // Initialize host vectors
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // --- 3. Get Device Pointers to Mapped Memory ---
  float *d_a, *d_b, *d_c;
  checkCudaError(cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0),
                 "cudaHostGetDevicePointer for d_a");
  checkCudaError(cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0),
                 "cudaHostGetDevicePointer for d_b");
  checkCudaError(cudaHostGetDevicePointer((void **)&d_c, (void *)h_c, 0),
                 "cudaHostGetDevicePointer for d_c");

  std::cout << "Device pointers obtained." << std::endl;

  // --- 4. Launch the Kernel ---
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Launching kernel..." << std::endl;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  // --- 5. Synchronize ---
  checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  std::cout << "Kernel execution complete." << std::endl;

  // --- 6. Verify the Results on the Host ---
  bool success = true;
  for (int i = 0; i < n; ++i) {
    if (h_c[i] != (h_a[i] + h_b[i])) {
      std::cerr << "Verification failed at index " << i << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Verification successful!" << std::endl;
  }

  // --- 7. Free Memory ---
  checkCudaError(cudaFreeHost(h_a), "cudaFreeHost for h_a");
  checkCudaError(cudaFreeHost(h_b), "cudaFreeHost for h_b");
  checkCudaError(cudaFreeHost(h_c), "cudaFreeHost for h_c");
  std::cout << "Mapped pinned host memory freed." << std::endl;

  return 0;
}

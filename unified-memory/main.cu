#include <chrono>
#include <iostream>
#include <vector>

// A standard CUDA error-checking macro
#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__     \
                << ": " << cudaGetErrorString(err_) << std::endl;              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Simple RAII Timer
class CudaTimer {
public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaEventRecord(start_));
  }
  ~CudaTimer() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    std::cout << "Elapsed time: " << ms << " ms" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }

private:
  cudaEvent_t start_, stop_;
};

// Kernel to perform one iteration of the Jacobi algorithm
__global__ void jacobi_kernel(const float *A, float *Anew, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary checks
  if (x > 0 && x < width - 1 && y > 0 && y < width - 1) {
    int idx = y * width + x;
    Anew[idx] =
        0.25f * (A[idx - 1] + A[idx + 1] + A[idx - width] + A[idx + width]);
  }
}

// --- Method 1: Traditional Manual Memory Management ---
void runManual(int width, int iters) {
  std::cout << "\n--- Running Manual (cudaMalloc/cudaMemcpy) Method ---"
            << std::endl;
  CudaTimer timer;

  size_t size = width * width * sizeof(float);
  std::vector<float> h_A(width * width);
  // Initialize with boundary conditions (e.g., hot top wall)
  for (int i = 0; i < width; ++i)
    h_A[i] = 100.0f;

  float *d_A, *d_Anew;
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_Anew, size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Anew, d_A, size, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

  for (int i = 0; i < iters; ++i) {
    jacobi_kernel<<<grid, block>>>(d_A, d_Anew, width);
    // Swap pointers for the next iteration
    float *temp = d_A;
    d_A = d_Anew;
    d_Anew = temp;
  }

  CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size, cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_Anew);
  // h_A now contains the result.
}

// --- Method 2: Naive Unified Memory (No Prefetching) ---
void runUnifiedMemory(int width, int iters) {
  std::cout << "\n--- Running Naive Unified Memory Method ---" << std::endl;
  CudaTimer timer;

  size_t size = width * width * sizeof(float);
  float *A, *Anew;
  CUDA_CHECK(cudaMallocManaged(&A, size));
  CUDA_CHECK(cudaMallocManaged(&Anew, size));

  // CPU accesses the data, forcing migration to Host RAM
  for (int i = 0; i < width * width; ++i)
    A[i] = 0.0f;
  for (int i = 0; i < width; ++i)
    A[i] = 100.0f;
  for (int i = 0; i < width * width; ++i)
    Anew[i] = A[i];

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

  for (int i = 0; i < iters; ++i) {
    // GPU accesses the data, causing on-demand page faulting and migration
    jacobi_kernel<<<grid, block>>>(A, Anew, width);
    float *temp = A;
    A = Anew;
    Anew = temp;
  }

  // This sync is needed before the CPU can safely access the result
  CUDA_CHECK(cudaDeviceSynchronize());
  // CPU reads result, causing migration back to host.
  std::cout << "Result at center (naive UM): "
            << A[(width / 2) * width + (width / 2)] << std::endl;

  cudaFree(A);
  cudaFree(Anew);
}

// --- Method 3: Optimized Unified Memory (With Prefetching) ---
void runUnifiedMemoryWithPrefetching(int width, int iters) {
  std::cout << "\n--- Running Optimized Unified Memory (Prefetching) Method ---"
            << std::endl;
  CudaTimer timer;

  size_t size = width * width * sizeof(float);
  float *A, *Anew;
  CUDA_CHECK(cudaMallocManaged(&A, size));
  CUDA_CHECK(cudaMallocManaged(&Anew, size));

  // CPU accesses data
  for (int i = 0; i < width * width; ++i)
    A[i] = 0.0f;
  for (int i = 0; i < width; ++i)
    A[i] = 100.0f;
  for (int i = 0; i < width * width; ++i)
    Anew[i] = A[i];

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

  for (int i = 0; i < iters; ++i) {
    // Proactively move data to the GPU before the kernel needs it.
    // This is like a non-blocking cudaMemcpy.
    CUDA_CHECK(cudaMemPrefetchAsync(A, size, device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(Anew, size, device, 0));

    jacobi_kernel<<<grid, block>>>(A, Anew, width);

    float *temp = A;
    A = Anew;
    Anew = temp;
  }

  // Proactively move the final result back to the CPU
  CUDA_CHECK(cudaMemPrefetchAsync(A, size, cudaCpuDeviceId, 0));

  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Result at center (prefetch UM): "
            << A[(width / 2) * width + (width / 2)] << std::endl;

  cudaFree(A);
  cudaFree(Anew);
}

int main() {
  const int WIDTH = 2048;
  const int ITERATIONS = 100;

  std::cout << "Grid Dimensions: " << WIDTH << "x" << WIDTH << std::endl;
  std::cout << "Iterations: " << ITERATIONS << std::endl;

  runManual(WIDTH, ITERATIONS);
  runUnifiedMemory(WIDTH, ITERATIONS);
  runUnifiedMemoryWithPrefetching(WIDTH, ITERATIONS);

  return 0;
}

#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <numeric>
#include <random>
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

// ==========================================================================
// KERNEL 1: Naive approach using only global memory atomics
// ==========================================================================
// This kernel is simple but can be slow if many threads try to update the
// same bin simultaneously, as all atomics are serialized at the L2 cache.
__global__ void histogram_naive_kernel(const unsigned int *data,
                                       unsigned int *bins, size_t data_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // Use a grid-stride loop to allow a fixed number of threads
  // to process an arbitrarily large dataset.
  for (int i = tid; i < data_size; i += stride) {
    unsigned int bin_index = data[i];

    // This is a global atomic operation. If many threads have the same
    // bin_index, they will all contend for the same memory location,
    // and the hardware will force them to execute one-by-one.
    atomicAdd(&bins[bin_index], 1);
  }
}

// ==========================================================================
// KERNEL 2: Optimized approach using shared memory atomics
// ==========================================================================
// This kernel is the standard high-performance pattern. It greatly reduces
// contention on global memory by first creating a private histogram for
// each block in fast __shared__ memory.
__global__ void histogram_shared_mem_kernel(const unsigned int *data,
                                            unsigned int *bins,
                                            size_t data_size, int num_bins) {

  // Allocate a histogram in shared memory.
  // It must be declared as extern so its size can be set at launch time.
  extern __shared__ unsigned int s_bins[];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // --- Phase 1: Initialize the shared memory histogram to zero ---
  // Each thread in the block helps to zero out a portion of the shared
  // histogram.
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    s_bins[i] = 0;
  }
  // // parallel initialize sm
  // s_bins[threadIdx.x] = 0;

  // Synchronize to ensure all threads in the block have finished
  // zeroing out the shared memory before proceeding.
  __syncthreads();

  // --- Phase 2: Compute the partial histogram in shared memory ---
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = gridDim.x * blockDim.x;

  for (int i = tid; i < data_size; i += stride) {
    unsigned int bin_index = data[i];

    // This is a shared memory atomic. It is MUCH faster than a global
    // atomic because it is resolved within the SM and doesn't need
    // to go to the L2 cache or main VRAM. Contention is still possible
    // but it's resolved much more quickly.
    atomicAdd(&s_bins[bin_index], 1);
  }

  // Synchronize to ensure all threads in the block have finished
  // contributing to the shared histogram before writing to global memory.
  __syncthreads();

  // --- Phase 3: Add the partial results to the global histogram ---
  // Each thread adds one of the partial results from the shared histogram
  // to the final global histogram.
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    if (s_bins[i] > 0) {
      // Now we perform the global atomic, but we've reduced the number
      // of global atomic operations from data_size to (num_blocks * num_bins).
      // Contention is massively reduced.
      atomicAdd(&bins[i], s_bins[i]);
    }
  }
}

int main() {
  // --- 1. Configuration ---
  const size_t DATA_SIZE = 1 << 24; // ~16.7 million elements
  const int NUM_BINS = 256;
  const int BLOCK_SIZE = 256;

  std::cout << "Data size: " << DATA_SIZE << " elements" << std::endl;
  std::cout << "Number of bins: " << NUM_BINS << std::endl;
  std::cout << "Block size: " << BLOCK_SIZE << std::endl;

  // --- 2. Host Data Initialization ---
  std::vector<unsigned int> h_data(DATA_SIZE);

  std::mt19937 rng(42); // Seeded for reproducibility
  std::uniform_int_distribution<unsigned int> dist(0, NUM_BINS - 1);
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    h_data[i] = dist(rng);
  }

  std::vector<unsigned int> h_bins_cpu(NUM_BINS, 0);
  std::vector<unsigned int> h_bins_gpu_naive(NUM_BINS, 0);
  std::vector<unsigned int> h_bins_gpu_shared(NUM_BINS, 0);

  // --- 3. Device Memory Allocation ---
  unsigned int *d_data, *d_bins;
  CUDA_CHECK(cudaMalloc(&d_data, DATA_SIZE * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));

  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  int grid_size = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // --- CUDA Event Setup ---
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float ms_naive = 0.0f, ms_shared = 0.0f;

  // --- 5. Execute Naive Global Atomic Kernel ---
  std::cout << "\nRunning naive global atomic kernel..." << std::endl;
  CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));

  CUDA_CHECK(cudaEventRecord(start));
  histogram_naive_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, d_bins, DATA_SIZE);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

  CUDA_CHECK(cudaMemcpy(h_bins_gpu_naive.data(), d_bins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  // --- 6. Execute Optimized Shared Memory Kernel ---
  std::cout << "Running optimized shared memory kernel..." << std::endl;
  CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
  size_t shared_mem_size = NUM_BINS * sizeof(unsigned int);

  CUDA_CHECK(cudaEventRecord(start));
  histogram_shared_mem_kernel<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(
      d_data, d_bins, DATA_SIZE, NUM_BINS);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

  CUDA_CHECK(cudaMemcpy(h_bins_gpu_shared.data(), d_bins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  // --- CPU Histogram Timing ---
  std::cout << "\nRunning CPU histogram..." << std::endl;
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    h_bins_cpu[h_data[i]]++;
  }
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  double ms_cpu =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  // --- 7. Verification ---
  std::cout << "\nVerifying results..." << std::endl;
  bool success = true;
  for (int i = 0; i < NUM_BINS; ++i) {
    if (h_bins_cpu[i] != h_bins_gpu_naive[i] ||
        h_bins_cpu[i] != h_bins_gpu_shared[i]) {
      std::cerr << "Verification FAILED at bin " << i << "! "
                << "CPU: " << h_bins_cpu[i]
                << ", GPU Naive: " << h_bins_gpu_naive[i]
                << ", GPU Shared: " << h_bins_gpu_shared[i] << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Verification PASSED!" << std::endl;
  }

  // --- Print Timing Results ---
  std::cout << "\nTiming Results:" << std::endl;
  std::cout << "CPU histogram: " << ms_cpu << " ms" << std::endl;
  std::cout << "GPU naive kernel: " << ms_naive << " ms" << std::endl;
  std::cout << "GPU shared memory kernel: " << ms_shared << " ms" << std::endl;

  // --- 8. Cleanup ---
  cudaFree(d_data);
  cudaFree(d_bins);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::cout << "\nCleanup complete." << std::endl;

  return 0;
}

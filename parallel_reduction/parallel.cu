#include <algorithm> // For std::min
#include <cmath>
#include <iostream>
#include <numeric> // For std::accumulate
#include <vector>

// Helper for checking CUDA errors
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =================================================================================
// Reduction Helper & Kernels
// =================================================================================
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Kernel to produce partial sums (one per block)
// Used by Method 1 (CPU Reduction) and Method 3 (Multi-Kernel)
__global__ void producePartialSums(const float *g_in, float *g_out, size_t n) {
  extern __shared__ float s_mem[];

  unsigned int tid = threadIdx.x;
  unsigned int block_stride = gridDim.x * blockDim.x;

  float my_sum = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < n; i += block_stride) {
    my_sum += g_in[i];
  }

  my_sum = warpReduceSum(my_sum);

  unsigned int warp_id = tid / 32;
  if ((tid % 32) == 0) {
    s_mem[warp_id] = my_sum;
  }

  __syncthreads();

  if (tid < (blockDim.x / 32)) {
    my_sum = s_mem[tid];
  } else {
    my_sum = 0.0f;
  }

  if (warp_id == 0) {
    my_sum = warpReduceSum(my_sum);
    if (tid == 0) {
      g_out[blockIdx.x] = my_sum;
    }
  }
}

// __global__ void produceFinalSums(const float *g_in, float *g_out, size_t n) {
//   unsigned int tid = threadIdx.x;
//   unsigned int block_stride = gridDim.x * blockDim.x;
//
//   if(tid+block_stride<gridDim.x){
//   }
// }

// Kernel for Method 2: uses atomicAdd for the final step
__global__ void atomicReduce(const float *g_in, float *g_out_final, size_t n) {
  extern __shared__ float s_mem[];

  unsigned int tid = threadIdx.x;
  unsigned int block_stride = gridDim.x * blockDim.x;

  // Each block calculates its partial sum exactly as before
  float my_sum = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < n; i += block_stride) {
    my_sum += g_in[i];
  }
  my_sum = warpReduceSum(my_sum);
  unsigned int warp_id = tid / 32;
  if ((tid % 32) == 0) {
    s_mem[warp_id] = my_sum;
  }
  __syncthreads();
  if (tid < (blockDim.x / 32)) {
    my_sum = s_mem[tid];
  } else {
    my_sum = 0.0f;
  }
  if (warp_id == 0) {
    my_sum = warpReduceSum(my_sum);
    if (tid == 0) {
      // <<< KEY CHANGE >>>
      // Instead of writing to g_out[blockIdx.x], add to a single location
      // atomically
      atomicAdd(&g_out_final[0], my_sum);
    }
  }
}

int main() {
  // --- Setup ---
  size_t n = 1 << 26; // ~67 million elements
  size_t data_size = n * sizeof(float);
  int threads_per_block = 1024;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  num_blocks = std::min(num_blocks, 4096);

  std::cout << "Input Data Size: " << n << " floats ("
            << data_size / (1024 * 1024) << " MB)" << std::endl;
  std::cout << "Grid Size: " << num_blocks << " blocks, " << threads_per_block
            << " threads per block" << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Allocate host memory
  std::vector<float> h_in(n);
  double cpu_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    h_in[i] = (i % 256) * 0.01f;
    cpu_sum += h_in[i];
  }

  // Allocate device memory
  float *d_in, *d_partial_out, *d_final_out;
  CHECK_CUDA(cudaMalloc(&d_in, data_size));
  CHECK_CUDA(cudaMalloc(&d_partial_out, num_blocks * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_final_out, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), data_size, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  float kernel_time, transfer_time, total_time;

  // --- Method 1: CPU Reduction ---
  std::cout << "Method 1: GPU Partial Sum + CPU Final Sum" << std::endl;
  CHECK_CUDA(cudaEventRecord(start));
  producePartialSums<<<num_blocks, threads_per_block,
                       (threads_per_block / 32) * sizeof(float)>>>(
      d_in, d_partial_out, n);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&kernel_time, start, stop));

  std::vector<float> h_partial_out(num_blocks);
  CHECK_CUDA(cudaEventRecord(start));
  CHECK_CUDA(cudaMemcpy(h_partial_out.data(), d_partial_out,
                        num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&transfer_time, start, stop));

  double gpu_sum_cpu_reduce =
      std::accumulate(h_partial_out.begin(), h_partial_out.end(), 0.0);
  total_time = kernel_time + transfer_time;
  printf("  Kernel Time:   %8.4f ms\n", kernel_time);
  printf("  Copy Time:     %8.4f ms\n", transfer_time);
  printf("  Total Time:    %8.4f ms\n", total_time);
  printf("  Result: %.1f (CPU Reference: %.1f)\n", gpu_sum_cpu_reduce, cpu_sum);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // --- Method 2: Atomic Operation Reduction ---
  std::cout << "Method 2: Single Kernel with Atomic Reduction" << std::endl;
  CHECK_CUDA(cudaMemset(
      d_final_out, 0, sizeof(float))); // Important: Zero out the sum location!
  CHECK_CUDA(cudaEventRecord(start));
  atomicReduce<<<num_blocks, threads_per_block,
                 (threads_per_block / 32) * sizeof(float)>>>(d_in, d_final_out,
                                                             n);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));

  float gpu_sum_atomic;
  CHECK_CUDA(cudaMemcpy(&gpu_sum_atomic, d_final_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("  Total Time:    %8.4f ms\n", total_time);
  printf("  Result: %.1f (CPU Reference: %.1f)\n", (double)gpu_sum_atomic,
         cpu_sum);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // --- Method 3: Multi-Kernel GPU Reduction ---
  std::cout << "Method 3: Multi-Kernel GPU Reduction" << std::endl;
  CHECK_CUDA(cudaEventRecord(start));
  // Kernel 1: Same as in Method 1
  producePartialSums<<<num_blocks, threads_per_block,
                       (threads_per_block / 32) * sizeof(float)>>>(
      d_in, d_partial_out, n);
  // Kernel 2: Reduce the partial sums with a single block
  producePartialSums<<<(num_blocks + threads_per_block - 1) / threads_per_block,
                       threads_per_block,
                       (threads_per_block / 32) * sizeof(float)>>>(
      d_partial_out, d_final_out, num_blocks);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));

  float gpu_sum_multikernel;
  CHECK_CUDA(cudaMemcpy(&gpu_sum_multikernel, d_final_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("  Total Time:    %8.4f ms\n", total_time);
  printf("  Result: %.1f (CPU Reference: %.1f)\n", (double)gpu_sum_multikernel,
         cpu_sum);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // --- Cleanup ---
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_partial_out));
  CHECK_CUDA(cudaFree(d_final_out));

  return 0;
}

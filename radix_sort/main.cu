#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// Enhanced error checking macro with more context
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error in %s at line %d: %s\n", __FILE__,    \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    err = cudaDeviceSynchronize();                                             \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Synchronization Error in %s at line %d: %s\n",     \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// We use a 4-bit radix, so there are 16 possible "digit" values (bins).
constexpr int RADIX_BITS = 4;
constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 16

// Phase 1: Each block computes a local histogram of the 4-bit keys for its tile
// of data.
__global__ void histogram_kernel(const unsigned int *in,
                                 unsigned int *histograms, int n, int pass) {
  // Use fast shared memory for each block's private counters
  extern __shared__ unsigned int s_counts[];

  int tid = threadIdx.x;
  int block_id = blockIdx.x;
  int block_dim = blockDim.x;

  // Have threads cooperate to clear shared memory
  if (tid < RADIX_SIZE) {
    s_counts[tid] = 0;
  }
  __syncthreads();

  int shift = pass * RADIX_BITS;
  int elements_per_thread =
      (n + (block_dim * gridDim.x) - 1) / (block_dim * gridDim.x);
  int start_idx = (block_id * block_dim + tid) * elements_per_thread;
  int end_idx = min(start_idx + elements_per_thread, n);

  // Each thread processes its assigned elements
  for (int i = start_idx; i < end_idx; ++i) {
    // Extract the 4-bit "digit" for the current pass
    unsigned int key = (in[i] >> shift) & (RADIX_SIZE - 1);
    atomicAdd(&s_counts[key], 1);
  }
  __syncthreads();

  // Each thread writes one bin count from shared memory to the global histogram
  // array
  if (tid < RADIX_SIZE) {
    histograms[block_id * RADIX_SIZE + tid] = s_counts[tid];
  }
}

// Phase 2: Each block re-reads its tile and scatters elements to their sorted
// positions.
__global__ void scatter_kernel(const unsigned int *in, unsigned int *out,
                               const unsigned int *offsets, int n, int pass) {
  // Use shared memory to cache the starting positions for this block
  extern __shared__ unsigned int s_offsets[];

  int tid = threadIdx.x;
  int block_id = blockIdx.x;
  int block_dim = blockDim.x;

  // Each block loads its relevant section of the global prefix sum table into
  // shared memory
  if (tid < RADIX_SIZE) {
    s_offsets[tid] = offsets[block_id * RADIX_SIZE + tid];
  }
  __syncthreads();

  int shift = pass * RADIX_BITS;
  int elements_per_thread =
      (n + (block_dim * gridDim.x) - 1) / (block_dim * gridDim.x);
  int start_idx = (block_id * block_dim + tid) * elements_per_thread;
  int end_idx = min(start_idx + elements_per_thread, n);

  // Each thread processes its elements again
  for (int i = start_idx; i < end_idx; ++i) {
    unsigned int key = (in[i] >> shift) & (RADIX_SIZE - 1);

    // Atomically get and increment the position for this key. This gives us the
    // final sorted index.
    unsigned int dest_idx = atomicAdd(&s_offsets[key], 1);
    out[dest_idx] = in[i];
  }
}

// --- Host-side Orchestration ---

void radix_sort(unsigned int *d_in, int n) {
  int threads = 256;
  int blocks = std::min((n + (threads * 4) - 1) / (threads * 4), 1024);

  unsigned int *d_out;
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(unsigned int)));

  unsigned int *d_histograms;
  CHECK_CUDA(
      cudaMalloc(&d_histograms, blocks * RADIX_SIZE * sizeof(unsigned int)));

  unsigned int *d_scan_values;
  CHECK_CUDA(
      cudaMalloc(&d_scan_values, blocks * RADIX_SIZE * sizeof(unsigned int)));

  std::vector<unsigned int> h_histograms(blocks * RADIX_SIZE);

  unsigned int *current_in = d_in;
  unsigned int *current_out = d_out;

  // 32-bit integers, 4 bits per pass -> 8 passes
  for (int pass = 0; pass < 32 / RADIX_BITS; ++pass) {
    // Launch histogram kernel and check for errors
    histogram_kernel<<<blocks, threads, RADIX_SIZE * sizeof(unsigned int)>>>(
        current_in, d_histograms, n, pass);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // NOTE: For simplicity, this prefix sum is done on the host. For maximum
    // performance, this would be its own highly-optimized parallel scan kernel.
    CHECK_CUDA(cudaMemcpy(h_histograms.data(), d_histograms,
                          h_histograms.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned int> h_scan_values(h_histograms.size());

    unsigned int global_sums[RADIX_SIZE] = {0};
    for (int i = 0; i < blocks; ++i) {
      for (int j = 0; j < RADIX_SIZE; ++j) {
        global_sums[j] += h_histograms[i * RADIX_SIZE + j];
      }
    }

    unsigned int global_offsets[RADIX_SIZE] = {0};
    for (int i = 1; i < RADIX_SIZE; ++i) {
      global_offsets[i] = global_offsets[i - 1] + global_sums[i - 1];
    }

    std::vector<unsigned int> per_bin_offsets(RADIX_SIZE, 0);
    for (int i = 0; i < blocks; ++i) {
      for (int j = 0; j < RADIX_SIZE; ++j) {
        h_scan_values[i * RADIX_SIZE + j] =
            global_offsets[j] + per_bin_offsets[j];
        per_bin_offsets[j] += h_histograms[i * RADIX_SIZE + j];
      }
    }

    CHECK_CUDA(cudaMemcpy(d_scan_values, h_scan_values.data(),
                          h_scan_values.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Launch scatter kernel and check for errors
    scatter_kernel<<<blocks, threads, RADIX_SIZE * sizeof(unsigned int)>>>(
        current_in, current_out, d_scan_values, n, pass);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::swap(current_in, current_out);
  }

  if (current_in != d_in) {
    CHECK_CUDA(cudaMemcpy(d_in, d_in, n * sizeof(unsigned int),
                          cudaMemcpyDeviceToDevice));
  }

  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_histograms));
  CHECK_CUDA(cudaFree(d_scan_values));

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
  const int N = 1024 * 1024 * 16; // 16 Million integers

  std::vector<unsigned int> h_in(N);
  std::vector<unsigned int> h_out(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dis;
  for (int i = 0; i < N; ++i) {
    h_in[i] = dis(gen);
  }

  unsigned int *d_data;
  CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(unsigned int)));

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(d_data, h_in.data(), N * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::cout << "Starting CUDA Radix Sort for " << N << " elements..."
            << std::endl;
  CHECK_CUDA(cudaEventRecord(start));

  radix_sort(d_data, N);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "  -> CUDA Sort Time: " << ms << " ms" << std::endl;

  CHECK_CUDA(cudaMemcpy(h_out.data(), d_data, N * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  std::cout << "Verifying against std::sort..." << std::endl;
  std::sort(h_in.begin(), h_in.end());

  bool success = true;
  for (long long i = 0; i < N; ++i) {
    if (h_in[i] != h_out[i]) {
      std::cerr << "  -> Mismatch at index " << i << ": expected " << h_in[i]
                << ", got " << h_out[i] << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "  -> Verification PASSED!" << std::endl;
  } else {
    std::cout << "  -> Verification FAILED!" << std::endl;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_data));

  return 0;
}

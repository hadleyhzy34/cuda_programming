#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  {                                                                             \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1);                                                                  \
    }                                                                           \
  }

// Using 8-bit radix (256 bins) is usually more efficient than 4-bit
// for large datasets as it halves the number of passes (4 passes for 32-bit).
constexpr int RADIX_BITS = 8;
constexpr int RADIX_SIZE = 1 << RADIX_BITS;

// Phase 1: Histogram
__global__ void histogram_kernel(const unsigned int* in,
                                 unsigned int* histograms,
                                 int n,
                                 int shift) {
  extern __shared__ unsigned int s_counts[];

  // Initialize shared memory
  for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) {
    s_counts[i] = 0;
  }
  __syncthreads();

  // Coalesced memory access: threads process data in a stride
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    unsigned int bin = (in[i] >> shift) & (RADIX_SIZE - 1);
    atomicAdd(&s_counts[bin], 1);
  }
  __syncthreads();

  // Export shared memory to global histogram
  if (threadIdx.x < RADIX_SIZE) {
    histograms[blockIdx.x * RADIX_SIZE + threadIdx.x] = s_counts[threadIdx.x];
  }
}

// Phase 2: Scatter (Stable version)
__global__ void scatter_kernel(const unsigned int* in,
                               unsigned int* out,
                               const unsigned int* offsets, // Per-block, per-bin offsets
                               int n,
                               int shift) {
  // Shared memory for a block's chunk of data and for local prefix sum.
  extern __shared__ unsigned int s_data[];
  const int block_size = 256; // Assumes blockDim.x is 256

  // Carve up shared memory
  unsigned int* s_val = s_data;                               // size: block_size
  unsigned int* s_bin = &s_val[block_size];                   // size: block_size
  unsigned int* s_rank = &s_bin[block_size];                  // size: block_size
  unsigned int* s_local_hist = &s_rank[block_size];           // size: RADIX_SIZE
  unsigned int* s_local_offsets = &s_local_hist[RADIX_SIZE];  // size: RADIX_SIZE

  // Process input in chunks of size gridDim.x * blockDim.x
  for (int chunk_base = 0; chunk_base < n; chunk_base += gridDim.x * block_size) {
    int idx = chunk_base + blockIdx.x * block_size + threadIdx.x;

    // 1. Initialize local histogram
    if (threadIdx.x < RADIX_SIZE) {
      s_local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // 2. Load chunk and build local histogram
    if (idx < n) {
      unsigned int val = in[idx];
      s_val[threadIdx.x] = val;
      unsigned int bin = (val >> shift) & (RADIX_SIZE - 1);
      s_bin[threadIdx.x] = bin;
      atomicAdd(&s_local_hist[bin], 1);
    }
    __syncthreads();

    // 3. Local exclusive prefix sum on the histogram
    if (threadIdx.x < RADIX_SIZE) {
      unsigned int rank = 0;
      for (int i = 0; i < threadIdx.x; ++i) {
        rank += s_local_hist[i];
      }
      s_local_offsets[threadIdx.x] = rank;
    }
    __syncthreads();

    // 4. Calculate rank within the chunk and scatter
    if (idx < n) {
      unsigned int bin = s_bin[threadIdx.x];
      unsigned int local_rank = atomicAdd(&s_local_offsets[bin], 1);
      unsigned int global_offset = offsets[(chunk_base/block_size + blockIdx.x) * RADIX_SIZE + bin];
      
      out[global_offset + local_rank] = s_val[threadIdx.x];
    }
    __syncthreads(); // Ensure all writes are done before next chunk
  }
}

void radix_sort(unsigned int* d_data, int n) {
  int threads = 256;
  // Limit blocks to keep the histogram table manageable
  int blocks = std::min((n + threads - 1) / threads, 512);

  unsigned int *d_out, *d_hist;
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&d_hist, blocks * RADIX_SIZE * sizeof(unsigned int)));

  // We need to calculate total blocks for the scatter kernel's grid-stride loop
  int total_scatter_blocks = (n + threads - 1) / threads;

  std::vector<unsigned int> h_hist(total_scatter_blocks * RADIX_SIZE);
  std::vector<unsigned int> h_offsets(total_scatter_blocks * RADIX_SIZE);

  unsigned int* d_in_ptr = d_data;
  unsigned int* d_out_ptr = d_out;

  for (int shift = 0; shift < 32; shift += RADIX_BITS) {
    // 1. Histogram
    histogram_kernel<<<total_scatter_blocks, threads, RADIX_SIZE * sizeof(unsigned int)>>>(d_in_ptr,
                                                                                           d_hist,
                                                                                           n,
                                                                                           shift);

    // 2. Prefix Sum (Host-side for simplicity)
    CHECK_CUDA(cudaMemcpy(h_hist.data(),
                          d_hist,
                          h_hist.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    unsigned int total = 0;
    for (int bin = 0; bin < RADIX_SIZE; ++bin) {
      for (int b = 0; b < total_scatter_blocks; ++b) {
        h_offsets[b * RADIX_SIZE + bin] = total;
        total += h_hist[b * RADIX_SIZE + bin];
      }
    }

    CHECK_CUDA(cudaMemcpy(d_hist,
                          h_offsets.data(),
                          h_offsets.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // 3. Scatter
    // Shared mem: val[256] + bin[256] + rank[256] + hist[256] + offsets[256]
    size_t shared_mem_size = (threads * 3 + RADIX_SIZE * 2) * sizeof(unsigned int);
    scatter_kernel<<<blocks, threads, shared_mem_size>>>(d_in_ptr,
                                                         d_out_ptr,
                                                         d_hist,
                                                         n,
                                                         shift);
    // view results for every step
    std::vector<unsigned int> temp(n);

    CHECK_CUDA(
        cudaMemcpy(temp.data(), d_out_ptr, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
      std::cout << temp[i] << " ";
    }
    std::cout << std::endl;

    // Swap buffers
    std::swap(d_in_ptr, d_out_ptr);
  }

  // If final data is in d_out, copy it back to d_data
  if (d_in_ptr != d_data) {
    CHECK_CUDA(cudaMemcpy(d_data, d_in_ptr, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }

  cudaFree(d_out);
  cudaFree(d_hist);
}

int main() {
  // const int N = 1024 * 1024;  // 1M elements
  const int N = 10 * 10;  // 100 elements
  std::vector<unsigned int> h_in(N);
  std::vector<unsigned int> h_temp(N);

  // std::generate(h_in.begin(), h_in.end(), []() { return std::rand(); });

  std::generate(h_in.begin(), h_in.end(), []() { return std::rand() % 1000; });

  for (auto i = 0; i < N; i++) {
    std::cout << h_in[i] << std::endl;
  }

  unsigned int* d_data;
  CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(unsigned int)));
  CHECK_CUDA(cudaMemcpy(d_data, h_in.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice));

  radix_sort(d_data, N);

  std::vector<unsigned int> h_out(N);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  std::cout << "output of h_out is: " << std::endl;
  for (auto i = 0; i < N; i++) {
    std::cout << h_out[i] << " ";
  }
  std::cout << std::endl;

  std::sort(h_in.begin(), h_in.end());
  bool match = true;
  for (int i = 0; i < N; i++)
    if (h_in[i] != h_out[i]) {
      match = false;
      break;
    }

  std::cout << (match ? "Success!" : "Failure!") << std::endl;

  cudaFree(d_data);
  return 0;
}

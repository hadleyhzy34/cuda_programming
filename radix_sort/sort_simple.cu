#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#define CHECK(call)                                                        \
  {                                                                        \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      printf("CUDA Error: %s at %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1);                                                             \
    }                                                                      \
  }

// 1-bit radix = 2 bins (Bin 0 and Bin 1)
const int BINS = 2;

// Phase 1: Count 0s and 1s in this block's data
__global__ void histogram_1bit(unsigned int* data,
                               int* block_counts,
                               int n,
                               int bit  // number of bits to shift
) {
  __shared__ int s_counts[BINS];
  if (threadIdx.x < BINS)
    s_counts[threadIdx.x] = 0;
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Extract the bit at the current position (0 or 1)
    int bin = (data[idx] >> bit) & 1;
    atomicAdd(&s_counts[bin], 1);
  }
  __syncthreads();

  // block_counts[blockID,binID] = sm[binID]
  if (threadIdx.x < BINS) {
    block_counts[blockIdx.x * BINS + threadIdx.x] = s_counts[threadIdx.x];
  }
}

// Phase 2: Move data to d_out based on pre-calculated offsets
__global__ void scatter_1bit(unsigned int* d_in,
                             unsigned int* d_out,
                             int* d_offsets,
                             int n,
                             int bit  // number of bits to shift
) {
  __shared__ int s_offsets[BINS];
  if (threadIdx.x < BINS) {
    s_offsets[threadIdx.x] = d_offsets[blockIdx.x * BINS + threadIdx.x];
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // if (idx < n) {
  //   unsigned int val = d_in[idx];
  //   int bin = (val >> bit) & 1;
  //
  //   // Find the specific position for this element
  //   // atomicAdd returns the OLD value, then increments it
  //   int dest = atomicAdd(&s_offsets[bin], 1);
  //   d_out[dest] = val;
  // }

  // assignment to each element within same block must be in order
  if (idx < n && threadIdx.x == 0) {
    for (auto i = 0; i < blockDim.x && i + idx < n; i++) {
      unsigned int val = d_in[idx + i];
      int bin = (val >> bit) & 1;

      // Find the specific position for this element
      // atomicAdd returns the OLD value, then increments it
      int dest = atomicAdd(&s_offsets[bin], 1);
      d_out[dest] = val;
      printf("idx:%d,i:%d,dest:%d,d_out[dest]: %f\n", idx, i, dest, d_out[dest]);
    }
  }
}

void radix_sort_simple(unsigned int* d_data, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  unsigned int* d_temp;
  int* d_block_counts;
  CHECK(cudaMalloc(&d_temp, n * sizeof(unsigned int)));
  CHECK(cudaMalloc(&d_block_counts, blocks * BINS * sizeof(int)));

  std::vector<int> h_counts(blocks * BINS);
  std::vector<int> h_offsets(blocks * BINS);

  unsigned int* src = d_data;
  unsigned int* dst = d_temp;

  // We process the integer bit-by-bit (0 to 31)
  for (int bit = 0; bit < 32; bit++) {
    // 1. Get Histogram
    histogram_1bit<<<blocks, threads>>>(src, d_block_counts, n, bit);

    // 2. Calculate Offsets on Host
    CHECK(cudaMemcpy(h_counts.data(),
                     d_block_counts,
                     blocks * BINS * sizeof(int),
                     cudaMemcpyDeviceToHost));

    int total_zeros = 0;
    for (int b = 0; b < blocks; b++) {
      total_zeros += h_counts[b * BINS + 0];
    }

    int current_zero_offset = 0;
    int current_one_offset = total_zeros;  // 1s start after all 0s
                                           //
    std::cout << "current number of 0s: " << current_zero_offset << std::endl;

    for (int b = 0; b < blocks; b++) {
      h_offsets[b * BINS + 0] = current_zero_offset;
      h_offsets[b * BINS + 1] = current_one_offset;

      current_zero_offset += h_counts[b * BINS + 0];
      current_one_offset += h_counts[b * BINS + 1];
    }

    CHECK(cudaMemcpy(d_block_counts,
                     h_offsets.data(),
                     blocks * BINS * sizeof(int),
                     cudaMemcpyHostToDevice));

    // 3. Scatter elements into dst
    scatter_1bit<<<blocks, threads>>>(src, dst, d_block_counts, n, bit);

    // 4. Swap src and dst for the next bit
    std::swap(src, dst);
  }

  // If we ended up with the result in d_temp, copy it back to d_data
  if (src != d_data) {
    CHECK(cudaMemcpy(d_data, src, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }

  cudaFree(d_temp);
  cudaFree(d_block_counts);
}

int main() {
  // const int N = 1000000;
  const int N = 1e6;
  std::vector<unsigned int> h_data(N);
  for (int i = 0; i < N; i++)
    h_data[i] = rand() % 1000;

  unsigned int* d_data;
  CHECK(cudaMalloc(&d_data, N * sizeof(unsigned int)));
  CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice));

  radix_sort_simple(d_data, N);

  std::vector<unsigned int> h_result(N);
  CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  for (auto i = 0; i < N; i++) {
    std::cout << h_result[i] << " ";
  }
  std::cout << std::endl;

  std::sort(h_data.begin(), h_data.end());
  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (h_data[i] != h_result[i]) {
      ok = false;
      break;
    }
  }

  std::cout << (ok ? "Sorted Correctly!" : "Error in Sorting") << std::endl;
  cudaFree(d_data);
  return 0;
}

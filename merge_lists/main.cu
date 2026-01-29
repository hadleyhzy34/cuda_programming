#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
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

// Device function to find the "split point" using binary search
// This finds how many elements from list A and list B should be
// included to reach the 'k-th' position in the merged output.
__device__ int get_split_point(const int* A, int nA, const int* B, int nB, int k) {
  int low = max(0, k - nB);
  int high = min(k, nA);

  while (low < high) {
    int i = low + (high - low) / 2;
    int j = k - i;
    if (A[i] < B[j - 1]) {
      low = i + 1;
    } else {
      high = i;
    }
  }
  return low;
}

__global__ void merge_kernel(const int* A, int nA, const int* B, int nB, int* C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalN = nA + nB;

  // Each thread processes a small chunk of the output
  int elements_per_thread = (totalN + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x);
  int k_start = idx * elements_per_thread;
  int k_end = min(k_start + elements_per_thread, totalN);

  if (k_start >= totalN)
    return;

  // Find where this thread starts in A and B
  int i_ptr = get_split_point(A, nA, B, nB, k_start);
  int j_ptr = k_start - i_ptr;

  // Sequential merge for the small chunk assigned to this thread
  for (int k = k_start; k < k_end; ++k) {
    if (i_ptr < nA && (j_ptr >= nB || A[i_ptr] <= B[j_ptr])) {
      C[k] = A[i_ptr++];
    } else {
      C[k] = B[j_ptr++];
    }
  }
}

// Helper to launch the kernel
void launch_merge(const int* d_A, int nA, const int* d_B, int nB, int* d_C) {
  int threads = 256;
  int blocks = 128;  // Adjust based on data size
  merge_kernel<<<blocks, threads>>>(d_A, nA, d_B, nB, d_C);
  CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
  int K = 8;                // Number of lists
  int N_per_list = 100000;  // Elements per list
  int total_elements = K * N_per_list;

  // 1. Prepare Data
  std::vector<int*> d_lists(K);
  // std::vector<int> h_data;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::cout << "Generating " << K << " sorted lists..." << std::endl;
  for (int i = 0; i < K; ++i) {
    std::vector<int> list(N_per_list);
    for (int& x : list)
      x = gen() % 1000000;
    std::sort(list.begin(), list.end());  // Pre-sort each list

    CHECK_CUDA(cudaMalloc(&d_lists[i], N_per_list * sizeof(int)));
    CHECK_CUDA(
        cudaMemcpy(d_lists[i], list.data(), N_per_list * sizeof(int), cudaMemcpyHostToDevice));
  }

  // 2. Parallel Reduction Merge
  // We use a "Tournament" approach.
  // Round 1: List 0+1 -> NewList, List 2+3 -> NewList...
  std::vector<int> current_sizes(K, N_per_list);
  int active_lists = K;

  while (active_lists > 1) {
    std::cout << "Merging " << active_lists << " lists..." << std::endl;
    int next_active = (active_lists + 1) / 2;
    std::vector<int*> next_lists(next_active);
    std::vector<int> next_sizes(next_active);

    for (int i = 0; i < active_lists; i += 2) {
      if (i + 1 < active_lists) {
        int new_size = current_sizes[i] + current_sizes[i + 1];
        CHECK_CUDA(cudaMalloc(&next_lists[i / 2], new_size * sizeof(int)));

        launch_merge(d_lists[i],
                     current_sizes[i],
                     d_lists[i + 1],
                     current_sizes[i + 1],
                     next_lists[i / 2]);

        next_sizes[i / 2] = new_size;
        cudaFree(d_lists[i]);
        cudaFree(d_lists[i + 1]);
      } else {
        // If odd number of lists, just carry the last one over
        next_lists[i / 2] = d_lists[i];
        next_sizes[i / 2] = current_sizes[i];
      }
    }
    d_lists = next_lists;
    current_sizes = next_sizes;
    active_lists = next_active;
  }

  // 3. Verification
  std::vector<int> h_result(total_elements);
  CHECK_CUDA(cudaMemcpy(h_result.data(),
                        d_lists[0],
                        total_elements * sizeof(int),
                        cudaMemcpyDeviceToHost));

  bool sorted = true;
  for (int i = 1; i < total_elements; ++i) {
    if (h_result[i] < h_result[i - 1]) {
      sorted = false;
      break;
    }
  }

  std::cout << (sorted ? "VERIFICATION PASSED!" : "VERIFICATION FAILED!") << std::endl;

  cudaFree(d_lists[0]);
  return 0;
}

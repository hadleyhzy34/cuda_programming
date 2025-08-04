#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256
// #define TILE_SIZE (2 * BLOCK_SIZE)
// #define TILE_SIZE 1024
// #define TILE_SIZE 256
// #define TILE_SIZE 128
#define TILE_SIZE 512

// === Utility: Check CUDA Error ===
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// === 1. CPU Single-threaded Merge ===
void cpu_merge(const int *A, int sizeA, const int *B, int sizeB, int *C) {
  int i = 0, j = 0, k = 0;
  while (i < sizeA && j < sizeB) {
    if (A[i] <= B[j])
      C[k++] = A[i++];
    else
      C[k++] = B[j++];
  }
  while (i < sizeA)
    C[k++] = A[i++];
  while (j < sizeB)
    C[k++] = B[j++];
}

// === 2. Naive CUDA Global Memory Merge ===
__global__ void gpu_merge_naive(const int *A, int sizeA, const int *B,
                                int sizeB, int *C) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = sizeA + sizeB;
  if (tid >= total)
    return;

  // Naively perform merge by binary search
  int low = max(0, tid - sizeB);
  int high = min(tid, sizeA);

  while (low < high) {
    int mid = (low + high) / 2;
    if (A[mid] <= B[tid - mid - 1])
      low = mid + 1;
    else
      high = mid;
  }

  int aIdx = low;
  int bIdx = tid - low;

  if (aIdx < sizeA && (bIdx >= sizeB || A[aIdx] <= B[bIdx]))
    C[tid] = A[aIdx];
  else
    C[tid] = B[bIdx];
}

// === Merge Path Utility ===
__device__ __forceinline__ void merge_path_search(int diag, const int *A,
                                                  int sizeA, const int *B,
                                                  int sizeB, int &a_idx,
                                                  int &b_idx) {
  // lower and upper boundary on number of elements from array A will assigned
  // to final C
  // total number of elements would be diag + 1
  // then rest of elements assigned to B will diag + 1 - mid
  // elements from B will be 0, ..., diag + 1 - mid - 2, diag + 1 - mid - 1
  int low = max(0, diag - sizeB);
  int high = min(diag, sizeA);

  // find last index of mid where A[mid] is largest value for final merged array
  // from 0 to mid
  while (low < high) {
    int mid = (low + high) / 2;
    int a_val = A[mid];
    int b_val = B[diag - mid - 1];
    if (a_val <= b_val)
      low = mid + 1;
    else
      high = mid;
  }

  a_idx = low;
  b_idx = diag - low;
}

// === 3. Shared Memory Merge using Merge Path ===
__global__ void gpu_merge_tile_shared(const int *A, int sizeA, const int *B,
                                      int sizeB, int *C) {
  __shared__ int sharedA[TILE_SIZE];
  __shared__ int sharedB[TILE_SIZE];

  int tid = threadIdx.x;
  int block_offset = blockIdx.x * TILE_SIZE;

  // std::cout<<"tid: "<<tid<<" bid: "<<blockIdx.x<<std::endl;
  // printf("tid: %d, bid: %d\n",tid,blockIdx.x);

  // Determine merge range
  int diag_start = block_offset;
  int diag_end = min(block_offset + TILE_SIZE, sizeA + sizeB);

  int a_start, b_start, a_end, b_end;
  merge_path_search(diag_start, A, sizeA, B, sizeB, a_start, b_start);
  merge_path_search(diag_end, A, sizeA, B, sizeB, a_end, b_end);

  int a_len = a_end - a_start;
  int b_len = b_end - b_start;

  // // Serially Load into shared memory
  // for (int i = tid; i < a_len; i += blockDim.x)
  //     if (a_start + i < sizeA) sharedA[i] = A[a_start + i];
  // for (int i = tid; i < b_len; i += blockDim.x)
  //     if (b_start + i < sizeB) sharedB[i] = B[b_start + i];

  // parallel load into shared memory
  if (a_start + tid < sizeA)
    sharedA[tid] = A[a_start + tid];
  if (b_start + tid < sizeB)
    sharedB[tid] = B[b_start + tid];

  __syncthreads();

  // Merge sharedA and sharedB into global C
  int total = a_len + b_len;
  if (tid < total) {
    int ai, bi;
    merge_path_search(tid, sharedA, a_len, sharedB, b_len, ai, bi);

    int a_val = (ai < a_len) ? sharedA[ai] : INT_MAX;
    int b_val = (bi < b_len) ? sharedB[bi] : INT_MAX;

    int val = (a_val <= b_val) ? a_val : b_val;
    if (block_offset + tid < sizeA + sizeB)
      C[block_offset + tid] = val;
  }
  // for (int i = tid; i < total; i += blockDim.x) {
  //     int ai, bi;
  //     merge_path_search(i, sharedA, a_len, sharedB, b_len, ai, bi);

  //     int a_val = (ai < a_len) ? sharedA[ai] : INT_MAX;
  //     int b_val = (bi < b_len) ? sharedB[bi] : INT_MAX;

  //     int val = (a_val <= b_val) ? a_val : b_val;
  //     if (block_offset + i < sizeA + sizeB)
  //         C[block_offset + i] = val;
  // }
}

// === Benchmark Utilities ===
float benchmark_gpu_merge(int mode, const int *d_A, int sizeA, const int *d_B,
                          int sizeB, int *d_C) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int total = sizeA + sizeB;

  // *** THE CRITICAL ADDITION ***
  // Clear the output buffer before timing and launching the kernel.
  // Set it to a value like -1 to make errors obvious.
  CHECK_CUDA(cudaMemset(d_C, 0xFF, total * sizeof(int))); // Fills with -1

  CHECK_CUDA(cudaEventRecord(start));
  if (mode == 1) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gpu_merge_naive<<<blocks, threads>>>(d_A, sizeA, d_B, sizeB, d_C);
  } else if (mode == 2) {
    int tiles = (total + TILE_SIZE - 1) / TILE_SIZE;
    // int tiles = (total + TILE_SIZE - 1) / BLOCK_SIZE;
    // gpu_merge_tile_shared<<<tiles, BLOCK_SIZE>>>(d_A, sizeA, d_B, sizeB,
    // d_C); gpu_merge_tile_shared<<<tiles, 1>>>(d_A, sizeA, d_B, sizeB, d_C);
    gpu_merge_tile_shared<<<tiles, TILE_SIZE>>>(d_A, sizeA, d_B, sizeB, d_C);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

// === Main ===
int main() {
  int sizeA = 1 << 20; // 1M
  int sizeB = 1 << 20;
  // int sizeA = 1<<8;
  // int sizeB = 1<<8;

  std::vector<int> h_A(sizeA), h_B(sizeB), h_C(sizeA + sizeB),
      h_ref(sizeA + sizeB);

  // Fill A and B with sorted values
  for (int i = 0; i < sizeA; ++i)
    h_A[i] = i * 2;
  for (int i = 0; i < sizeB; ++i)
    h_B[i] = i * 2 + 1;

  // CPU baseline
  auto t1 = std::chrono::high_resolution_clock::now();
  cpu_merge(h_A.data(), sizeA, h_B.data(), sizeB, h_ref.data());
  auto t2 = std::chrono::high_resolution_clock::now();
  double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[CPU Merge]: " << cpu_time << " ms\n";

  // Allocate on device
  int *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_C, (sizeA + sizeB) * sizeof(int)));

  CHECK_CUDA(
      cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(int), cudaMemcpyHostToDevice));

  // Naive GPU Merge
  float t_naive = benchmark_gpu_merge(1, d_A, sizeA, d_B, sizeB, d_C);
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::cout << "[GPU Naive Merge]: " << t_naive << " ms\n";
  std::cout << "  Correct? " << (h_C == h_ref ? "Yes" : "No") << "\n";

  // Shared memory tile-based merge path
  float t_shared = benchmark_gpu_merge(2, d_A, sizeA, d_B, sizeB, d_C);
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (int i = 1; i < sizeA + sizeB; i++) {
    if (h_C[i] < h_C[i - 1]) {
      std::cout << "wrong sequence: " << std::endl;
      return 0;
    }
    if (h_C[i] == 0) {
      std::cout << i << std::endl;
    }
    // std::cout<<h_C[i]<<" ";
  }
  std::cout << "[GPU Shared Merge]: " << t_shared << " ms\n";
  std::cout << "  Correct? " << (h_C == h_ref ? "Yes" : "No") << "\n";

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  return 0;
}

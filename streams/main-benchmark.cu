#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernel to initialize data on the GPU
__global__ void init_data(float *data, int n, int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = static_cast<float>(idx + offset);
  }
}

// First processing kernel
__global__ void process_A(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = powf(data[idx], 0.75f);
  }
}

// Second processing kernel
__global__ void process_B(float *data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= factor;
  }
}

int main() {
  // --- Benchmark Parameters ---
  const int TOTAL_ELEMENTS = 32 * 1024 * 1024; // 32M floats
  const int NUM_CHUNKS = 64;
  const int CHUNK_ELEMENTS = TOTAL_ELEMENTS / NUM_CHUNKS;
  const size_t CHUNK_BYTES = CHUNK_ELEMENTS * sizeof(float);
  const int NUM_STREAMS = 3; // For pipelining
  const float PROCESS_FACTOR = 3.14f;

  std::cout << "--- Benchmark Setup ---" << std::endl;
  std::cout << "Total Elements: " << TOTAL_ELEMENTS << std::endl;
  std::cout << "Number of Chunks: " << NUM_CHUNKS << std::endl;
  std::cout << "Elements per Chunk: " << CHUNK_ELEMENTS << std::endl;
  std::cout << "Number of Streams (for Async): " << NUM_STREAMS << std::endl;
  std::cout << "-----------------------" << std::endl << std::endl;

  // --- Memory Allocation ---
  // Pageable host memory for the simple synchronous case
  std::vector<float> h_in_pageable(TOTAL_ELEMENTS);
  std::vector<float> h_out_pageable(TOTAL_ELEMENTS);
  std::iota(h_in_pageable.begin(), h_in_pageable.end(), 0.0f);

  // Pinned host memory for high-performance async transfers
  float *h_in_pinned, *h_out_pinned;
  CHECK_CUDA(cudaMallocHost(&h_in_pinned, TOTAL_ELEMENTS * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out_pinned, TOTAL_ELEMENTS * sizeof(float)));
  for (int i = 0; i < TOTAL_ELEMENTS; ++i)
    h_in_pinned[i] = static_cast<float>(i);

  // Device memory
  float *d_data;
  CHECK_CUDA(cudaMalloc(&d_data, CHUNK_BYTES));

  // --- CUDA Events for Timing ---
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  float ms;

  // Kernel launch configuration
  dim3 grid((CHUNK_ELEMENTS + 255) / 256, 1, 1);
  dim3 block(256, 1, 1);

  // ===================================================================
  // 1. Plain Synchronous Benchmark
  // ===================================================================
  std::cout << "Running Plain Synchronous Benchmark..." << std::endl;
  CHECK_CUDA(cudaEventRecord(start));

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    int offset = i * CHUNK_ELEMENTS;
    CHECK_CUDA(cudaMemcpy(d_data, h_in_pageable.data() + offset, CHUNK_BYTES,
                          cudaMemcpyHostToDevice));
    init_data<<<grid, block>>>(d_data, CHUNK_ELEMENTS, offset);
    process_A<<<grid, block>>>(d_data, CHUNK_ELEMENTS);
    process_B<<<grid, block>>>(d_data, PROCESS_FACTOR, CHUNK_ELEMENTS);
    CHECK_CUDA(cudaMemcpy(h_out_pageable.data() + offset, d_data, CHUNK_BYTES,
                          cudaMemcpyDeviceToHost));
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "  -> Result: " << ms << " ms" << std::endl << std::endl;

  // ===================================================================
  // 2. Multi-Stream Asynchronous Benchmark
  // ===================================================================
  std::cout << "Running Multi-Stream Asynchronous Benchmark..." << std::endl;
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; ++i)
    CHECK_CUDA(cudaStreamCreate(&streams[i]));

  CHECK_CUDA(cudaEventRecord(start));

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    int stream_idx = i % NUM_STREAMS;
    int offset = i * CHUNK_ELEMENTS;

    // Enqueue all work for this chunk on its assigned stream
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_in_pinned + offset, CHUNK_BYTES,
                               cudaMemcpyHostToDevice, streams[stream_idx]));
    init_data<<<grid, block, 0, streams[stream_idx]>>>(d_data, CHUNK_ELEMENTS,
                                                       offset);
    process_A<<<grid, block, 0, streams[stream_idx]>>>(d_data, CHUNK_ELEMENTS);
    process_B<<<grid, block, 0, streams[stream_idx]>>>(d_data, PROCESS_FACTOR,
                                                       CHUNK_ELEMENTS);
    CHECK_CUDA(cudaMemcpyAsync(h_out_pinned + offset, d_data, CHUNK_BYTES,
                               cudaMemcpyDeviceToHost, streams[stream_idx]));
  }

  // Wait for all streams to complete all their work
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "  -> Result: " << ms << " ms" << std::endl << std::endl;

  for (int i = 0; i < NUM_STREAMS; ++i)
    CHECK_CUDA(cudaStreamDestroy(streams[i]));

  // ===================================================================
  // 3. CUDA Graph Benchmark
  // ===================================================================
  std::cout << "Running CUDA Graph Benchmark..." << std::endl;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t capture_stream;
  CHECK_CUDA(cudaStreamCreate(&capture_stream));

  // --- Capture Phase ---
  // Capture the sequence of operations for a single chunk.
  CHECK_CUDA(
      cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));

  CHECK_CUDA(cudaMemcpyAsync(d_data, h_in_pinned, CHUNK_BYTES,
                             cudaMemcpyHostToDevice, capture_stream));
  init_data<<<grid, block, 0, capture_stream>>>(d_data, CHUNK_ELEMENTS, 0);
  process_A<<<grid, block, 0, capture_stream>>>(d_data, CHUNK_ELEMENTS);
  process_B<<<grid, block, 0, capture_stream>>>(d_data, PROCESS_FACTOR,
                                                CHUNK_ELEMENTS);
  CHECK_CUDA(cudaMemcpyAsync(h_out_pinned, d_data, CHUNK_BYTES,
                             cudaMemcpyDeviceToHost, capture_stream));

  CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph));

  // --- Instantiate Phase ---
  CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  CHECK_CUDA(cudaEventRecord(start));

  // --- Launch Phase ---
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    // NOTE: A real implementation would update the input/output pointers here
    // using cudaGraphExecUpdate for each chunk. For this benchmark, we
    // repeatedly process the same chunk to focus on the launch overhead
    // comparison.
    CHECK_CUDA(cudaGraphLaunch(instance, capture_stream));
  }

  CHECK_CUDA(cudaStreamSynchronize(capture_stream));
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "  -> Result: " << ms << " ms" << std::endl << std::endl;

  // --- Cleanup ---
  CHECK_CUDA(cudaGraphExecDestroy(instance));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaStreamDestroy(capture_stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaFreeHost(h_in_pinned));
  CHECK_CUDA(cudaFreeHost(h_out_pinned));

  return 0;
}

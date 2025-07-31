#include <iostream>
#include <vector>

// CUDA error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void process_data(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // A simple workload
        data[idx] = sqrt(pow(data[idx], 1.5f));
    }
}

int main() {
    int n_elements = 32 * 1024 * 1024; // 32M elements
    size_t bytes = n_elements * sizeof(float);
    const int num_streams = 2;

    // --- Host Memory ---
    // Pageable memory for the synchronous version
    float* h_data_pageable = new float[n_elements];
    // Pinned memory for the asynchronous version
    float* h_data_pinned;
    CHECK_CUDA(cudaMallocHost(&h_data_pinned, bytes));

    // Initialize data
    for (int i = 0; i < n_elements; ++i) {
        h_data_pageable[i] = static_cast<float>(i);
        h_data_pinned[i] = static_cast<float>(i);
    }
    
    // --- Device Memory ---
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // CUDA Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // --- 1. Synchronous Approach (No Streams, Pageable Memory) ---
    std::cout << "Starting synchronous version..." << std::endl;
    CHECK_CUDA(cudaEventRecord(start));
    
    CHECK_CUDA(cudaMemcpy(d_data, h_data_pageable, bytes, cudaMemcpyHostToDevice));
    process_data<<<(n_elements + 255) / 256, 256>>>(d_data, n_elements);
    CHECK_CUDA(cudaMemcpy(h_data_pageable, d_data, bytes, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Synchronous version took: " << ms << " ms" << std::endl;


    // --- 2. Asynchronous Approach (2 Streams, Pinned Memory) ---
    std::cout << "\nStarting asynchronous version with " << num_streams << " streams..." << std::endl;
    
    // Create streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    int chunk_size = n_elements / num_streams;
    size_t chunk_bytes = chunk_size * sizeof(float);
    dim3 grid((chunk_size + 255) / 256, 1, 1);
    dim3 block(256, 1, 1);
    
    CHECK_CUDA(cudaEventRecord(start));

    // This loop orchestrates the overlap
    for (int i = 0; i < num_streams; ++i) {
        int offset = i * chunk_size;
        // Asynchronously copy a chunk of data IN the current stream
        CHECK_CUDA(cudaMemcpyAsync(d_data + offset, h_data_pinned + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        
        // Asynchronously launch kernel on that chunk IN the same stream
        process_data<<<grid, block, 0, streams[i]>>>(d_data + offset, chunk_size);
        
        // Asynchronously copy the result back IN the same stream
        CHECK_CUDA(cudaMemcpyAsync(h_data_pinned + offset, d_data + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Wait for ALL streams to finish their work
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Asynchronous version took: " << ms << " ms" << std::endl;


    // --- Cleanup ---
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFreeHost(h_data_pinned));
    delete[] h_data_pageable;

    return 0;
}

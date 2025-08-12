#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <numeric>

// --- CUDA Utilities ---
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(1); \
    } \
}

// --- Kernel 1: Naive Global Memory Merge ---

// __device__ function for binary search on global memory
__device__ int global_binary_search(const int* arr, int n, int val) {
    int low = 0;
    int high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] < val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__global__ void global_merge_kernel(const int* A, int nA, const int* B, int nB, int* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one element from A
    if (idx < nA) {
        int val = A[idx];
        int rank = idx + global_binary_search(B, nB, val);
        C[rank] = val;
    }

    // Each thread also handles one element from B
    if (idx < nB) {
        int val = B[idx];
        int rank = idx + global_binary_search(A, nA, val);
        C[rank] = val;
    }
}

// --- Kernel 2: Tiled Shared Memory Merge ---

#define TILE_SIZE 256 // Must be equal to block size for this implementation

// __device__ function for binary search on __shared__ memory
__device__ int shared_binary_search(const int* s_arr, int n, int val) {
    int low = 0;
    int high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (s_arr[mid] < val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__global__ void tiled_merge_kernel(const int* A, int nA, const int* B, int nB, int* C) {
    // This kernel performs a local merge. Each block merges one tile from A and one from B.
    // The final output C will not be fully sorted, but a sequence of sorted chunks.
    // This is designed to benchmark the core merge operation at the block level.

    __shared__ int sA[TILE_SIZE];
    __shared__ int sB[TILE_SIZE];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    // Pointers to the start of the tiles in global memory
    const int* a_tile_ptr = A + block_id * TILE_SIZE;
    const int* b_tile_ptr = B + block_id * TILE_SIZE;
    int* c_tile_ptr = C + block_id * 2 * TILE_SIZE;

    // --- 1. Cooperative Load from Global to Shared Memory ---
    // Each thread loads one element from each tile.
    if (block_id * TILE_SIZE + tid < nA) {
        sA[tid] = a_tile_ptr[tid];
    } else {
        sA[tid] = INT_MAX;
    }

    if (block_id * TILE_SIZE + tid < nB) {
        sB[tid] = b_tile_ptr[tid];
    } else {
        sB[tid] = INT_MAX;
    }
    __syncthreads(); // Ensure all data is loaded before merging

    // --- 2. Local Merge using Shared Memory ---
    // Each thread finds the final position for one element from sA and one from sB
    // within the local 2*TILE_SIZE output chunk.
    int val_a = sA[tid];
    int val_b = sB[tid];

    // Find rank of sA[tid] within the sB tile
    int rank_a = tid + shared_binary_search(sB, TILE_SIZE, val_a);

    // Find rank of sB[tid] within the sA tile
    int rank_b = tid + shared_binary_search(sA, TILE_SIZE, val_b);

    // --- 3. Write result to the correct location in the global output tile ---
    if (val_a != INT_MAX) {
        c_tile_ptr[rank_a] = val_a;
    }
    if (val_b != INT_MAX) {
        c_tile_ptr[rank_b] = val_b;
    }
}

// --- Host Code ---

// Verification for the global merge (expects one fully sorted array)
void verify_full_merge(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c, const std::string& kernel_name) {
    std::vector<int> expected_c;
    expected_c.resize(a.size() + b.size());
    std::merge(a.begin(), a.end(), b.begin(), b.end(), expected_c.begin());

    bool success = (c == expected_c);

    if (success) {
        std::cout << "[" << kernel_name << "] Verification PASSED!" << std::endl;
    } else {
        std::cout << "[" << kernel_name << "] Verification FAILED!" << std::endl;
    }
}

// Verification for the tiled merge (verifies each chunk is sorted)
void verify_tiled_merge(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c, int tile_size, const std::string& kernel_name) {
    bool success = true;
    int num_tiles = (a.size() + tile_size - 1) / tile_size;
    
    for (int i = 0; i < num_tiles; ++i) {
        std::vector<int> tile_a, tile_b, expected_chunk;
        
        // Extract the input tiles for this chunk
        int start_idx_a = i * tile_size;
        for(int j=0; j<tile_size && start_idx_a + j < a.size(); ++j) tile_a.push_back(a[start_idx_a+j]);
        int start_idx_b = i * tile_size;
        for(int j=0; j<tile_size && start_idx_b + j < b.size(); ++j) tile_b.push_back(b[start_idx_b+j]);
        
        // Merge them to get the expected result for this chunk
        expected_chunk.resize(tile_a.size() + tile_b.size());
        std::merge(tile_a.begin(), tile_a.end(), tile_b.begin(), tile_b.end(), expected_chunk.begin());

        // Compare with the actual chunk from GPU output
        int start_idx_c = i * 2 * tile_size;
        for(size_t j=0; j<expected_chunk.size(); ++j) {
            if (c[start_idx_c + j] != expected_chunk[j]) {
                success = false;
                break;
            }
        }
        if (!success) break;
    }

    if (success) {
        std::cout << "[" << kernel_name << "] Verification PASSED!" << std::endl;
    } else {
        std::cout << "[" << kernel_name << "] Verification FAILED!" << std::endl;
    }
}


int main() {
    // --- 1. Setup Host Data ---
    int nA = 1 << 22; // ~4 million elements
    int nB = 1 << 22;
    std::vector<int> h_A(nA), h_B(nB);
    
    // Fill with sorted, interleaved data
    for (int i = 0; i < nA; ++i) h_A[i] = i * 2;
    for (int i = 0; i < nB; ++i) h_B[i] = i * 2 + 1;

    std::cout << "Input Array Sizes: nA = " << nA << ", nB = " << nB << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // --- 2. Allocate Device Memory ---
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, (nA + nB) * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), nA * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), nB * sizeof(int), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- 3. Benchmark Global Memory Kernel ---
    {
        std::cout << "--- Running Global Memory Benchmark ---" << std::endl;
        int threadsPerBlock = 256;
        int numThreads = std::max(nA, nB);
        int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        std::vector<int> h_C_global(nA + nB);

        CUDA_CHECK(cudaMemset(d_C, 0, (nA + nB) * sizeof(int)));
        
        CUDA_CHECK(cudaEventRecord(start));
        global_merge_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, nA, d_B, nB, d_C);
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(h_C_global.data(), d_C, (nA + nB) * sizeof(int), cudaMemcpyDeviceToHost));
        
        double seconds = milliseconds / 1000.0;
        long long total_bytes = (long long)(nA + nB + nA + nB) * sizeof(int); // Read A, B, Write C
        double bandwidth = (double)total_bytes / (1e9 * seconds);

        std::cout.precision(3);
        std::cout << "Execution Time: " << std::fixed << milliseconds << " ms" << std::endl;
        std::cout << "Effective Bandwidth: " << std::fixed << bandwidth << " GB/s" << std::endl;
        
        verify_full_merge(h_A, h_B, h_C_global, "Global Kernel");
        std::cout << std::endl;
    }

    // --- 4. Benchmark Tiled Shared Memory Kernel ---
    {
        std::cout << "--- Running Tiled Shared Memory Benchmark ---" << std::endl;
        int threadsPerBlock = TILE_SIZE; // Must match TILE_SIZE
        int num_tiles_A = (nA + TILE_SIZE - 1) / TILE_SIZE;
        int num_tiles_B = (nB + TILE_SIZE - 1) / TILE_SIZE;
        int blocksPerGrid = std::max(num_tiles_A, num_tiles_B);
        std::vector<int> h_C_tiled(nA + nB);

        CUDA_CHECK(cudaMemset(d_C, 0, (nA + nB) * sizeof(int)));

        CUDA_CHECK(cudaEventRecord(start));
        tiled_merge_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, nA, d_B, nB, d_C);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, (nA + nB) * sizeof(int), cudaMemcpyDeviceToHost));

        double seconds = milliseconds / 1000.0;
        // Bandwidth calculation is the same, as we're performing the same logical operation
        long long total_bytes = (long long)(nA + nB + nA + nB) * sizeof(int);
        double bandwidth = (double)total_bytes / (1e9 * seconds);

        std::cout.precision(3);
        std::cout << "Execution Time: " << std::fixed << milliseconds << " ms" << std::endl;
        std::cout << "Effective Bandwidth: " << std::fixed << bandwidth << " GB/s" << std::endl;
        
        verify_tiled_merge(h_A, h_B, h_C_tiled, TILE_SIZE, "Tiled Kernel");
        std::cout << std::endl;
    }


    // --- 5. Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
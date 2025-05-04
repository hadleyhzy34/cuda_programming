#include <cuda_runtime.h>
#include <iostream>

__global__ void transposeNaive(float *__restrict__ out,
                               const float *__restrict__ in) {
  __shared__ float tile[32][32];
  int x = threadIdx.x;
  int y = threadIdx.y;

  // Load tile from global memory
  tile[y][x] = in[y + x * 32]; // Row-major access (no conflict)

  __syncthreads();

  // Store transposed tile to global memory
  out[y + x * 32] = tile[x][y]; // Column access (causes bank conflict)
}

__global__ void transposePadded(float *__restrict__ out,
                                const float *__restrict__ in) {
  __shared__ float tile[32][33]; // 33 columns (1 extra)
  int x = threadIdx.x;
  int y = threadIdx.y;

  // Load tile from global memory
  tile[y][x] = in[y + x * 32]; // Row-major access (no conflict)

  __syncthreads();

  // Store transposed tile to global memory
  out[y + x * 32] = tile[x][y]; // Column access (no conflict due to padding)
}

void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " failed: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  const int N = 32;
  const int size = N * N * sizeof(float);

  // Host buffers
  float *h_in = new float[N * N];
  float *h_out = new float[N * N];

  // Initialize input matrix
  for (int i = 0; i < N * N; ++i) {
    h_in[i] = i;
  }

  // Device buffers
  float *d_in, *d_out;
  checkCuda(cudaMalloc(&d_in, size), "cudaMalloc d_in");
  checkCuda(cudaMalloc(&d_out, size), "cudaMalloc d_out");

  // Copy input to device
  checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice),
            "cudaMemcpy d_in");

  // Events for timing
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Warm-up
  transposeNaive<<<1, dim3(32, 32)>>>(d_out, d_in);
  cudaDeviceSynchronize();

  // Measure naive kernel
  float ms_naive = 0.0f;
  for (int i = 0; i < 100; ++i) {
    cudaEventRecord(start);
    transposeNaive<<<1, dim3(32, 32)>>>(d_out, d_in);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms_naive += ms;
  }

  // Measure padded kernel
  float ms_padded = 0.0f;
  for (int i = 0; i < 100; ++i) {
    cudaEventRecord(start);
    transposePadded<<<1, dim3(32, 32)>>>(d_out, d_in);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms_padded += ms;
  }

  std::cout << "Naive kernel time: " << ms_naive << " ms" << std::endl;
  std::cout << "Padded kernel time: " << ms_padded << " ms" << std::endl;

  // Cleanup
  delete[] h_in;
  delete[] h_out;
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

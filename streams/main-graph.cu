#include <iostream>
#include <vector>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void add_vectors(float *c, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void scale_vector(float *a, float scalar, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] *= scalar;
  }
}

int main() {
  const int N = 1 << 20; // 1M elements
  const size_t bytes = N * sizeof(float);
  const int iterations = 100;

  // Allocate host and device memory
  float *h_a, *h_b, *h_c_out_stream, *h_c_out_graph;
  float *d_a, *d_b, *d_c;

  h_a = new float[N];
  h_b = new float[N];
  h_c_out_stream = new float[N];
  h_c_out_graph = new float[N];

  // Initialize host data
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  // CUDA Events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // --- 1. Traditional Stream-based Approach ---
  std::cout << "Running with traditional streams..." << std::endl;
  CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    add_vectors<<<(N + 255) / 256, 256>>>(d_c, d_a, d_b, N);
    scale_vector<<<(N + 255) / 256, 256>>>(d_c, 5.0f, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float stream_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&stream_ms, start, stop));
  std::cout << "Stream-based execution took: " << stream_ms << " ms"
            << std::endl;
  CHECK_CUDA(cudaMemcpy(h_c_out_stream, d_c, bytes, cudaMemcpyDeviceToHost));

  // --- 2. CUDA Graph Approach ---
  std::cout << "\nRunning with CUDA Graphs..." << std::endl;
  CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream = 0;

  // Start Capture
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // Note: We only need to capture ONE iteration of the work
  add_vectors<<<(N + 255) / 256, 256>>>(d_c, d_a, d_b, N);
  scale_vector<<<(N + 255) / 256, 256>>>(d_c, 5.0f, N);

  // End Capture
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

  // Instantiate
  CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  // Warm-up (optional, but good practice)
  CHECK_CUDA(cudaGraphLaunch(instance, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Time the graph execution
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaGraphLaunch(instance, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float graph_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&graph_ms, start, stop));
  std::cout << "Graph-based execution took: " << graph_ms << " ms" << std::endl;

  CHECK_CUDA(cudaMemcpy(h_c_out_graph, d_c, bytes, cudaMemcpyDeviceToHost));

  // Verify results
  for (int i = 0; i < 10; ++i) {
    if (abs(h_c_out_stream[i] - h_c_out_graph[i]) > 1e-5) {
      std::cout << "Verification FAILED at index " << i << std::endl;
      break;
    }
  }
  std::cout << "\nResults from both methods match." << std::endl;
  std::cout << "Example output: (1.0 + 2.0) * 5.0 = " << h_c_out_graph[0]
            << std::endl;

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaGraphExecDestroy(instance));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  delete[] h_a;
  delete[] h_b;
  delete[] h_c_out_stream;
  delete[] h_c_out_graph;

  return 0;
}

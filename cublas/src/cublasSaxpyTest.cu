#include <cstdlib> // For rand()
#include <iomanip> // For std::fixed, std::setprecision
#include <iostream>
#include <vector>

// CUDA includes
#include <cublas_v2.h> // The cuBLAS V2 API
#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__;            \
      std::cerr << ": " << cudaGetErrorString(err_) << std::endl;              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// cuBLAS Error Checking Macro
#define CUBLAS_CHECK(status)                                                   \
  {                                                                            \
    cublasStatus_t stat_ = (status);                                           \
    if (stat_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__;          \
      /* Note: cuBLAS doesn't have a direct equivalent to cudaGetErrorString   \
       * for all statuses */                                                   \
      /* For more detailed error messages, you might need to consult the       \
       * documentation */                                                      \
      /* or map specific statuses to strings yourself. */                      \
      std::cerr << ": Status code " << stat_ << std::endl;                     \
      /* A common one: if (stat_ == CUBLAS_STATUS_NOT_INITIALIZED) std::cerr   \
       * << " (CUBLAS_STATUS_NOT_INITIALIZED)" << std::endl; */                \
      /* ... add more specific status checks if needed */                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Helper function to print a matrix (assuming row-major C-style storage)
void print_matrix_row_major(const char *name, const float *matrix, int rows,
                            int cols) {
  std::cout << name << " (" << rows << "x" << cols
            << ") (Row-Major C-style):\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::fixed << std::setprecision(2) << matrix[i * cols + j]
                << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Helper function to print a vector
void print_vector(const char *name, const float *vec, int n) {
  std::cout << name << " (" << n << "):\n";
  for (int i = 0; i < n; ++i) {
    std::cout << std::fixed << std::setprecision(2) << vec[i] << "\t";
  }
  std::cout << "\n" << std::endl;
}

void run_saxpy_example(cublasHandle_t handle) {
  std::cout << "\n--- Running SAXPY Example (Y = a*X + Y) ---" << std::endl;
  int n = 5;
  float h_alpha = 2.0f;

  std::vector<float> h_X(n);
  std::vector<float> h_Y(n);
  std::vector<float> h_Y_expected(n);

  // Initialize host data
  for (int i = 0; i < n; ++i) {
    h_X[i] = static_cast<float>(i + 1);        // X = [1, 2, 3, 4, 5]
    h_Y[i] = static_cast<float>(10 * (i + 1)); // Y = [10, 20, 30, 40, 50]
    h_Y_expected[i] = h_alpha * h_X[i] + h_Y[i];
  }

  print_vector("Host X (initial):", h_X.data(), n);
  print_vector("Host Y (initial):", h_Y.data(), n);
  std::cout << "Host alpha: " << h_alpha << std::endl;

  float *d_X, *d_Y;
  CUDA_CHECK(cudaMalloc(&d_X, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, n * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_X, h_X.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_Y, h_Y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // SAXPY call
  // Pointer mode is HOST by default, so &h_alpha is fine
  CUBLAS_CHECK(cublasSaxpy(handle, n, &h_alpha, d_X, 1, d_Y, 1));

  std::vector<float> h_Y_result(n);
  CUDA_CHECK(cudaMemcpy(h_Y_result.data(), d_Y, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  print_vector("Host Y (GPU result):", h_Y_result.data(), n);
  print_vector("Host Y (expected):", h_Y_expected.data(), n);

  // Verify
  bool match = true;
  for (int i = 0; i < n; ++i) {
    if (std::abs(h_Y_result[i] - h_Y_expected[i]) > 1e-5) {
      match = false;
      break;
    }
  }
  std::cout << "SAXPY Result " << (match ? "MATCHES" : "DOES NOT MATCH")
            << " expected." << std::endl;

  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));
}

int main() {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  std::cout << "cuBLAS handle created." << std::endl;

  run_saxpy_example(handle);

  CUBLAS_CHECK(cublasDestroy(handle));
  std::cout << "cuBLAS handle destroyed." << std::endl;
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}

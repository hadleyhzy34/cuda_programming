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

int main() {
  // We'll fill this in
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  std::cout << "cuBLAS handle created." << std::endl;

  // ... your cuBLAS operations will go here ...

  CUBLAS_CHECK(cublasDestroy(handle));
  std::cout << "cuBLAS handle destroyed." << std::endl;
  CUDA_CHECK(cudaDeviceReset()); // Good practice to reset device at the end
  return 0;
}

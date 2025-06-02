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

// For C = alpha * A * B + beta * C
// (Matrices are typically column-major for cuBLAS, or you handle
// transpositions) Assuming row-major C = A * B, so A is M x K, B is K x N, C is
// M x N To use with cublasSgemm which expects column-major: C_col_major^T =
// B_col_major^T * A_col_major^T So, effectively, we compute C_transpose =
// B_transpose * A_transpose If A, B, C are already on GPU and in ROW-MAJOR
// order: A (M x K), B (K x N), C (M x N) cuBLAS sees B as the "first" matrix
// (KxN) and A as the "second" (MxK) if we want C_rm = A_rm * B_rm C_cm^T =
// B_cm^T * A_cm^T  (where _cm is column major) Let's assume A,B,C are on device
// in row-major. C(M,N) = A(M,K) * B(K,N) We can tell cuBLAS to treat B as op(B)
// and A as op(A) effectively. cublasSgemm(handle,
//             CUBLAS_OP_N,    // Transposition for B (second matrix in formula
//             B*A) CUBLAS_OP_N,    // Transposition for A (first matrix in
//             formula B*A) N,              // Rows of C (cols of B if no_trans,
//             rows of B if trans) M,              // Cols of C (cols of A if
//             no_trans, rows of A if trans) K,              // Common dimension
//             (cols of A if no_trans, rows of B if no_trans) &alpha,         //
//             Scalar alpha B_d,            // Pointer to B_d on device N, //
//             Leading dimension of B (ldb) A_d,            // Pointer to A_d on
//             device K,              // Leading dimension of A (lda) &beta, //
//             Scalar beta C_d,            // Pointer to C_d on device N); //
//             Leading dimension of C (ldc)
// This is a common source of confusion.
// Easier if you prepare data in column-major or mentally map.
// A_cm(K,M), B_cm(N,K), C_cm(N,M)
// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_d, M, B_d,
// K, &beta, C_d, M) if A(M,K), B(K,N), C(M,N) are already in column-major
// format on the device.

// If your data is ROW-MAJOR on device: A(M,K), B(K,N), C(M,N)
// To compute C = A*B (all row-major):
// C^T_cm = B^T_cm * A^T_cm
// So, effectively, cuBLAS computes D = X * Y where X=B^T, Y=A^T, D=C^T.
// The dimensions for Sgemm call:
// m for Sgemm = rows of X = N
// n for Sgemm = cols of Y = M
// k for Sgemm = cols of X / rows of Y = K
// X is B^T, so op(B) = CUBLAS_OP_T. Its original dimensions (K,N) give ldb=N
// (row-major). Y is A^T, so op(A) = CUBLAS_OP_T. Its original dimensions (M,K)
// give lda=K (row-major). D is C^T. Its original dimensions (M,N) give ldc=N
// (row-major).
float alpha = 1.0f;
float beta = 0.0f;
cublasStatus_t status = cublasSgemm(
    handle,
    CUBLAS_OP_T,    // Transpose B
    CUBLAS_OP_T,    // Transpose A
    N,              // Number of rows of matrix op(B) = N
    M,              // Number of columns of matrix op(A) = M
    K,              // Number of columns of op(B) / rows of op(A) = K
    &alpha, B_d, K, // B_d is KxN (row-major), ldb = K (cols of original B)
    A_d, M,         // A_d is MxK (row-major), lda = M (cols of original A)
    &beta, C_d, N); // C_d is MxN (row-major), ldc = N (cols of original C)
// After this, C_d contains A*B.
// Note: The lda, ldb, ldc parameters are the leading dimensions of the
// *original* matrices A, B, C as they are stored in memory (e.g., number of
// columns if row-major).

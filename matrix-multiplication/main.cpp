#include "src/naiveMM.hpp"
#include "src/sharedMM.hpp"
#include <iostream>

int main() {
  const int M = 512, N = 512, K = 512;
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C_naive = new float[M * N];
  float *h_C_shared = new float[M * N];

  // Initialize matrices with simple values
  for (int i = 0; i < M * K; ++i)
    h_A[i] = static_cast<float>(i % 10);
  for (int i = 0; i < K * N; ++i)
    h_B[i] = static_cast<float>(i % 5);

  std::cout << "Running naive version...\n";
  Computation::runNaive(h_A, h_B, h_C_naive, M, N, K);

  std::cout << "Running shared memory version...\n";
  Computation::runShared(h_A, h_B, h_C_shared, M, N, K);

  // Verify correctness (optional)
  bool correct = true;
  for (int i = 0; i < M * N; ++i) {
    if (std::abs(h_C_naive[i] - h_C_shared[i]) > 1e-5) {
      correct = false;
      break;
    }
  }
  std::cout << "Results match: " << (correct ? "Yes" : "No") << "\n";

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_naive;
  delete[] h_C_shared;
  return 0;
}

#include "src/dot_product.h"
#include <stdio.h>

int main() {
  const int N = 1024;
  const int bytes = N * sizeof(float);

  // Host arrays
  float *h_a = new float[N];
  float *h_b = new float[N];
  float h_c = 0.0f;

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)rand() % 100;
    h_b[i] = (float)rand() % 100;
  }

  computation::dotProduct(h_a, h_b, &h_c, N);

  // Verify result
  float cpu_result = 0.0f;
  for (int i = 0; i < N; i++) {
    cpu_result += h_a[i] * h_b[i];
  }

  printf("GPU Result: %f\n", h_c);
  printf("CPU Result: %f\n", cpu_result);

  // Cleanup
  delete[] h_a;
  delete[] h_b;

  return 0;
}

#include <cstdio>
#include <iostream>

__global__ void add(int *a, int *b, int *c) {
  int idx = threadIdx.x;
  c[idx] = a[idx] + b[idx];
}
int main() {
  const int Size = 10;
  int h_a[Size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int h_b[Size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int h_c[Size];

  int *d_a, *d_b, *d_c;
  // allocate memory on the device
  cudaMalloc((void **)&d_a, Size * sizeof(int));
  cudaMalloc((void **)&d_b, Size * sizeof(int));
  cudaMalloc((void **)&d_c, Size * sizeof(int));

  // copy data from host to device
  cudaMemcpy(d_a, h_a, 10 * sizeof(Size), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 10 * sizeof(Size), cudaMemcpyHostToDevice);

  add<<<1, Size>>>(d_a, d_b, d_c);

  // copy data from device to host
  cudaMemcpy(h_c, d_c, 10 * sizeof(Size), cudaMemcpyDeviceToHost);

  // test
  for (int i = 0; i < Size; i++) {
    printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
  }

  // free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

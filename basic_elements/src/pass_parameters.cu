#include <iostream>

__global__ void add(int a, int b, int *c) { *c = a + b; }

int main() {
  int h_c;

  int *d_c;

  // allocate memory for device data
  cudaMalloc((void **)&d_c, sizeof(int));

  add<<<1, 1>>>(1, 2, d_c);

  // copy data back to h_c
  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "h_c data value is: " << h_c << std::endl;
  return 0;
}

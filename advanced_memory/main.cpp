#include "src/unified_memory.cu"
#include <format>
#include <iostream>
#include <stdio.h>

template <typename T> void printTensor(T *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // std::cout << data[i] << " ";
    printf(std::format("value: {}\n", data[i]));
  }
  std::cout << std::endl;
}

int main() {
  int *data; // Unified Memory pointer
  size_t size = 1024;

  // Allocate Unified Memory
  cudaMallocManaged(&data, size * sizeof(int));

  // initialize data data
  memset(data, 0, size * sizeof(int));
  // // Initialize from CPU
  // for (int i = 0; i < 1024; i++)
  //   data[i] = 0;

  // Use in GPU kernel
  kernel<<<1, 1024>>>(data, size);
  cudaDeviceSynchronize();

  printTensor(data, size);

  cudaFree(data); // Free memory
  return 0;
}

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// #define N 2147483647
// #define N 65536

__global__ void add(int *a, int *b, int *c, int N) {
  // int tid = blockIdx.x + blockIdx.y * gridDim.x;
  int tid = blockIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

bool checkError(int idx, int a, int b, int c) {
  if (a + b != c) {
    std::cout << "at index " << idx << " answer is not correct" << std::endl;
    return false;
  }
  return true;
}

int main() {
  // size_t SIZE = SIZE_MAX;
  // size_t SIZE = INT_MAX;
  size_t SIZE = 1e8;
  printf("size is: %lu\n", SIZE);
  // const int size = 65535;
  // int h_a[SIZE];
  // int h_b[SIZE];
  // int h_c[SIZE];

  // heap array declaration
  int *h_a = (int *)malloc(SIZE * sizeof(int));
  int *h_b = (int *)malloc(SIZE * sizeof(int));
  int *h_c = (int *)malloc(SIZE * sizeof(int));

  printf("start writing data\n");
  for (int i = 0; i < SIZE; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }
  int *d_a, *d_b, *d_c;

  printf("start allocating cuda memory\n");
  // allocate memory for device data
  cudaMalloc((void **)&d_a, SIZE * sizeof(int));
  cudaMalloc((void **)&d_b, SIZE * sizeof(int));
  cudaMalloc((void **)&d_c, SIZE * sizeof(int));

  // copy from host to device
  cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  add<<<SIZE, 1>>>(d_a, d_b, d_c, SIZE);

  // copy data back to h_c
  cudaMemcpy(h_c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < SIZE; i++) {
    if (checkError(i, h_a[i], h_b[i], h_c[i])) {
      continue;
    } else {
      break;
    }
    // std::cout << h_c[i] << std::endl;
  }

  std::cout << "vectos addition complete and it's correct" << std::endl;
  // release memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

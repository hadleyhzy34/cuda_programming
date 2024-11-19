#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

typedef unsigned char uint8;
#define DIM 1000

// design complex data structure in cuda device
struct cuComplex {
  int r;
  int v;
  cuComplex() : r(0), v(0) {}
  cuComplex(int x, int y) : r(x), v(y) {}
  cuComplex *operator+(cuComplex *b) {
    cuComplex *c = new cuComplex();
    c->r = r + b->r;
    c->v = r + b->v;
    return c;
  }
  cuComplex *operator*(cuComplex *b) {
    cuComplex *c = new cuComplex();
    c->r = r * b->r - v * b->v;
    c->v = r * b->v + v * b->r;
    return c;
  }
};

__device__ int julia(int x, int y) {
  const float scale = 1.5;

  return 1;
}

__global__ void show(unsigned char *ptr) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = blockIdx.x + blockIdx.y * gridDim.x;
  // int tid = blockIdx.x + blockIdx.y * gridDim.x;

  int juliaValue = julia(x, y);
  ptr[DIM * DIM * 0 + offset] = 255 * juliaValue;
  ptr[DIM * DIM * 1 + offset] = 0;
  ptr[DIM * DIM * 2 + offset] = 0;
}

// write data to image
void write_image(const char *filename, uint8 *data, int width, int height) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    perror("failed to open file for writing");
    return;
  }

  // write width and height first, if needed for format consistency
  fwrite(&width, sizeof(int), 1, file);
  fwrite(&height, sizeof(int), 1, file);

  int image_size = width * height * 3;
  fwrite(data, sizeof(uint8), image_size, file);
  fclose(file);
}

int main() {
  // host data
  uint8 *ptr = (uint8 *)malloc(DIM * DIM * 3 * sizeof(uint8));

  // device data
  uint8 *dPtr;
  cudaMalloc((void **)&dPtr, DIM * DIM * 3 * sizeof(uint8));

  dim3 grid(DIM, DIM);
  show<<<grid, 1>>>(dPtr);

  // copy data from device to host
  cudaMemcpy(ptr, dPtr, DIM * DIM * 3 * sizeof(uint8), cudaMemcpyDeviceToHost);

  // write image
  write_image("test.ppm", ptr, DIM, DIM);
  // release memory
  cudaFree(dPtr);
  free(ptr);
  return 0;
}

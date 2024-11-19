#include <cuda_runtime.h>
#include <iostream>

#define DIM 1000 // Dimension of the output image
#define PI 3.1415926535897932f

struct cuComplex {
  float r;
  float i;

  __device__ cuComplex(float a, float b) : r(a), i(b) {}

  __device__ float magnitude2() const { return r * r + i * i; }

  __device__ cuComplex operator*(const cuComplex &a) const {
    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
  }

  __device__ cuComplex operator+(const cuComplex &a) const {
    return cuComplex(r + a.r, i + a.i);
  }
};

__device__ int julia(int x, int y) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
  float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);

  int i = 0;
  for (; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000)
      return 0;
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;

  // Now calculate the value at that position
  int juliaValue = julia(x, y);
  ptr[offset * 3 + 0] = 255 * juliaValue;
  ptr[offset * 3 + 1] = 0;
  ptr[offset * 3 + 2] = 0;
}

void save_image(const unsigned char *data, int width, int height,
                const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    std::cerr << "Could not open file for writing: " << filename << std::endl;
    return;
  }

  fprintf(f, "P6\n%d %d\n255\n", width, height);
  fwrite(data, 1, width * height * 3, f);
  fclose(f);
}

int main() {
  unsigned char *dev_bitmap;
  unsigned char *bitmap = new unsigned char[DIM * DIM * 3];

  // Allocate memory on the device
  cudaMalloc((void **)&dev_bitmap, DIM * DIM * 3);

  dim3 grid(DIM, DIM);
  kernel<<<grid, 1>>>(dev_bitmap);

  // Copy the bitmap back from the GPU to the CPU
  cudaMemcpy(bitmap, dev_bitmap, DIM * DIM * 3, cudaMemcpyDeviceToHost);

  // Save the image
  save_image(bitmap, DIM, DIM, "julia_set.ppm");

  // Free memory on GPU and CPU
  cudaFree(dev_bitmap);
  delete[] bitmap;

  std::cout << "Julia set image generated: julia_set.ppm" << std::endl;
  return 0;
}

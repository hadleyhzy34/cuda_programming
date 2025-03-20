#include <cuda_runtime.h>
#include <stdio.h>
#include <vector_functions.h>

// Sphere definition
struct Sphere {
  float radius;
  float3 position;
  float3 color;

  __device__ bool hit(const float3 &ray_origin, const float3 &ray_direction,
                      float &t) const {
    float3 oc = ray_origin - position;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0f * dot(oc, ray_direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
      float t1 = (-b - sqrtf(discriminant)) / (2.0f * a);
      float t2 = (-b + sqrtf(discriminant)) / (2.0f * a);

      t = (t1 > 0) ? t1 : t2;
      return t > 0;
    }
    return false;
  }
};

// CUDA kernel to render the scene
__global__ void render(float3 *output, int width, int height, Sphere *spheres,
                       int num_spheres) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  float3 ray_origin = make_float3(0, 0, 0);
  float3 ray_direction = normalize(make_float3(
      (x - width / 2.0f) / width, (y - height / 2.0f) / height, -1.0f));

  float3 pixel_color = make_float3(0, 0, 0);
  float t_min = 1e20f;

  for (int i = 0; i < num_spheres; i++) {
    float t = 0.0f;
    if (spheres[i].hit(ray_origin, ray_direction, t) && t < t_min) {
      t_min = t;
      pixel_color = spheres[i].color;
    }
  }

  output[idx] = pixel_color;
}

// Helper function for CUDA error checking
void checkCuda(cudaError_t result, const char *msg) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(result));
    exit(1);
  }
}

int main() {
  const int width = 800, height = 600;
  const int num_pixels = width * height;

  // Allocate host and device memory
  float3 *host_output = (float3 *)malloc(num_pixels * sizeof(float3));
  float3 *device_output;
  checkCuda(cudaMalloc(&device_output, num_pixels * sizeof(float3)),
            "Allocating device output");

  Sphere h_spheres[] = {
      {0.5f, make_float3(-0.5f, 0.0f, -3.0f), make_float3(1, 0, 0)},
      {0.3f, make_float3(0.5f, 0.0f, -2.0f), make_float3(0, 1, 0)},
  };
  Sphere *d_spheres;
  checkCuda(cudaMalloc(&d_spheres, sizeof(h_spheres)),
            "Allocating device spheres");
  checkCuda(cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres),
                       cudaMemcpyHostToDevice),
            "Copying spheres");

  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);
  render<<<blocks, threads>>>(device_output, width, height, d_spheres,
                              sizeof(h_spheres) / sizeof(Sphere));
  checkCuda(cudaDeviceSynchronize(), "Kernel execution");

  // Copy results back to host
  checkCuda(cudaMemcpy(host_output, device_output, num_pixels * sizeof(float3),
                       cudaMemcpyDeviceToHost),
            "Copying output");

  // Output as PPM (Portable Pixmap)
  FILE *file = fopen("output.ppm", "w");
  fprintf(file, "P3\n%d %d\n255\n", width, height);
  for (int i = 0; i < num_pixels; i++) {
    int r = (int)(host_output[i].x * 255.99f);
    int g = (int)(host_output[i].y * 255.99f);
    int b = (int)(host_output[i].z * 255.99f);
    fprintf(file, "%d %d %d\n", r, g, b);
  }
  fclose(file);

  // Cleanup
  free(host_output);
  cudaFree(device_output);
  cudaFree(d_spheres);

  printf("Rendering complete. Output saved to 'output.ppm'.\n");
  return 0;
}

#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

// 5x5 Gaussian kernel
__const__ float kernel[5][5] = {
    {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
    {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
    {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f},
    {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
    {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};

// Advanced: Gaussian blur using surface memory
__global__ void gaussianBlurSurface(cudaSurfaceObject_t input_surface,
                                    cudaSurfaceObject_t output_surface,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // printf("thread execution on %d,%d\n", x, y);

  // // 5x5 Gaussian kernel
  // const float kernel[5][5] = {
  //     {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
  //     {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
  //     {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f},
  //     {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
  //     {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};

  float sum = 0.0f;

  // Convolution with automatic boundary handling
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
        float pixel;
        // Surface automatically handles boundary conditions
        surf2Dread(&pixel, input_surface, (x + dx) * sizeof(float), y + dy);
        // sum += pixel * kernel[dy + 2][dx + 2];
        sum += pixel * 0.5;
      }
    }
  }

  surf2Dwrite(sum, output_surface, x * sizeof(float), y);
}

// Simple kernel to scale pixel values using surface memory
__global__ void scaleSurfaceKernel(cudaSurfaceObject_t input_surface,
                                   cudaSurfaceObject_t output_surface,
                                   int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  printf("handling thread: %d, %d\n", x, y);

  float value;
  surf2Dread(&value, input_surface, x * sizeof(float), y);
  surf2Dwrite(value * 2.0f, output_surface, x * sizeof(float), y);
}

int main() {
  const int width = 64;
  const int height = 64;
  std::cout << "Surface Memory Example: Scaling 64x64 image by 2\n";

  // Initialize host data
  std::vector<float> host_input(width * height, 0.0f);
  std::vector<float> host_output(width * height, 0.0f);
  for (int i = 0; i < width * height; i++) {
    host_input[i] = static_cast<float>(i % 100); // Simple pattern: 0 to 99
  }

  // Allocate CUDA arrays
  cudaArray_t input_array, output_array;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaError_t err;

  err = cudaMallocArray(&input_array, &channel_desc, width, height,
                        cudaArraySurfaceLoadStore);
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocArray failed for input: " << cudaGetErrorString(err)
              << "\n";
    return 1;
  }

  err = cudaMallocArray(&output_array, &channel_desc, width, height,
                        cudaArraySurfaceLoadStore);
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocArray failed for output: "
              << cudaGetErrorString(err) << "\n";
    cudaFreeArray(input_array);
    return 1;
  }

  // Create surface objects
  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypeArray;

  cudaSurfaceObject_t input_surface = 0, output_surface = 0;
  res_desc.res.array.array = input_array;
  err = cudaCreateSurfaceObject(&input_surface, &res_desc);
  if (err != cudaSuccess) {
    std::cerr << "cudaCreateSurfaceObject failed for input: "
              << cudaGetErrorString(err) << "\n";
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  res_desc.res.array.array = output_array;
  err = cudaCreateSurfaceObject(&output_surface, &res_desc);
  if (err != cudaSuccess) {
    std::cerr << "cudaCreateSurfaceObject failed for output: "
              << cudaGetErrorString(err) << "\n";
    cudaDestroySurfaceObject(input_surface);
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  // Copy input data to input array
  err = cudaMemcpy2DToArray(input_array, 0, 0, host_input.data(),
                            width * sizeof(float), width * sizeof(float),
                            height, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err)
              << "\n";
    cudaDestroySurfaceObject(input_surface);
    cudaDestroySurfaceObject(output_surface);
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  // scaleSurfaceKernel<<<grid, block>>>(input_surface, output_surface, width,
  //                                     height);

  gaussianBlurSurface<<<grid, block>>>(input_surface, output_surface, width,
                                       height);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    cudaDestroySurfaceObject(input_surface);
    cudaDestroySurfaceObject(output_surface);
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
    cudaDestroySurfaceObject(input_surface);
    cudaDestroySurfaceObject(output_surface);
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  // Copy output data back to host
  err = cudaMemcpy2DFromArray(host_output.data(), width * sizeof(float),
                              output_array, 0, 0, width * sizeof(float), height,
                              cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy2DFromArray failed: " << cudaGetErrorString(err)
              << "\n";
    cudaDestroySurfaceObject(input_surface);
    cudaDestroySurfaceObject(output_surface);
    cudaFreeArray(input_array);
    cudaFreeArray(output_array);
    return 1;
  }

  // Verify results
  bool success = true;
  for (int i = 0; i < width * height; i++) {
    if (fabs(host_output[i] - host_input[i] * 2.0f) > 1e-5) {
      std::cerr << "Verification failed at index " << i << ": expected "
                << host_input[i] * 2.0f << ", got " << host_output[i] << "\n";
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Surface memory example succeeded! Output verified.\n";
  }

  // Clean up
  cudaDestroySurfaceObject(input_surface);
  cudaDestroySurfaceObject(output_surface);
  cudaFreeArray(input_array);
  cudaFreeArray(output_array);

  return success ? 0 : 1;
}

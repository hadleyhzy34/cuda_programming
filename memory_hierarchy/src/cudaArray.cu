#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Error checking macro (updated for modern style)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":"      \
                << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernel to process the array using texture object
__global__ void processArray(cudaTextureObject_t texObj, float *output,
                             int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Read from texture object
    float value = tex2D<float>(texObj, x, y);
    // Double the value
    output[y * width + x] = value * 5.0f;
  }
}

int main() {
  const int width = 4;
  const int height = 4;

  // Host data
  std::vector<float> h_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                               7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f, 16.0f};

  // Step 1: Allocate CUDA array
  cudaArray_t cuArray;
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<float>(); // Simplified for single float
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

  // Step 2: Copy host data to CUDA array
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data.data(),
                                 width * sizeof(float), width * sizeof(float),
                                 height, cudaMemcpyHostToDevice));

  // Step 3: Create and configure texture object
  cudaResourceDesc resDesc{};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc{};
  texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp for x
  texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp for y
  texDesc.filterMode = cudaFilterModePoint;      // No interpolation
  texDesc.readMode = cudaReadModeElementType;    // Raw float values
  texDesc.normalizedCoords = 0;                  // Use absolute coordinates

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

  // Step 4: Allocate output device memory
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(float)));

  // Step 5: Launch kernel
  dim3 blockDim(2, 2);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);
  processArray<<<gridDim, blockDim>>>(texObj, d_output, width, height);
  CUDA_CHECK(cudaGetLastError()); // Check kernel launch
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 6: Copy results back to host
  std::vector<float> h_output(width * height);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                        width * height * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Print results
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::cout << h_output[y * width + x] << " ";
    }
    std::cout << std::endl;
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaDestroyTextureObject(texObj));
  CUDA_CHECK(cudaFreeArray(cuArray));

  return 0;
}

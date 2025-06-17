#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

#define WIDTH 2048
#define HEIGHT 2048

// Kernel using texture object (read-only)
__global__ void advectTextureKernel(cudaTextureObject_t texObj, float *output,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float value = tex2D<float>(texObj, x + 0.5f, y + 0.5f);
    output[y * width + x] = value;
  }
}

// Kernel using surface object (read & write)
__global__ void advectSurfaceKernel(cudaSurfaceObject_t surfObj, int width,
                                    int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float value;
    surf2Dread(&value, surfObj, x * sizeof(float), y);
    // For demonstration: write the same value back
    surf2Dwrite(value, surfObj, x * sizeof(float), y);
  }
}

void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "Error: " << msg << ": " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  size_t size = WIDTH * HEIGHT * sizeof(float);

  // Allocate host memory
  float *h_input = new float[WIDTH * HEIGHT];
  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    h_input[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_output;
  checkCuda(cudaMalloc(&d_output, size), "cudaMalloc d_output");

  // Create CUDA array
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  checkCuda(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT,
                            cudaArraySurfaceLoadStore),
            "cudaMallocArray");

  // Copy host data to CUDA array
  checkCuda(cudaMemcpy2DToArray(cuArray, 0, 0, h_input, WIDTH * sizeof(float),
                                WIDTH * sizeof(float), HEIGHT,
                                cudaMemcpyHostToDevice),
            "cudaMemcpy2DToArray");

  // Create texture object
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(texRes));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  checkCuda(cudaCreateTextureObject(&texObj, &texRes, &texDesc, nullptr),
            "cudaCreateTextureObject");

  // Create surface object
  cudaSurfaceObject_t surfObj = 0;
  cudaResourceDesc surfRes;
  memset(&surfRes, 0, sizeof(surfRes));
  surfRes.resType = cudaResourceTypeArray;
  surfRes.res.array.array = cuArray;
  checkCuda(cudaCreateSurfaceObject(&surfObj, &surfRes),
            "cudaCreateSurfaceObject");

  // Setup execution parameters
  dim3 threads(16, 16);
  dim3 blocks((WIDTH + threads.x - 1) / threads.x,
              (HEIGHT + threads.y - 1) / threads.y);

  // Warm-up runs
  advectTextureKernel<<<blocks, threads>>>(texObj, d_output, WIDTH, HEIGHT);
  cudaDeviceSynchronize();
  advectSurfaceKernel<<<blocks, threads>>>(surfObj, WIDTH, HEIGHT);
  cudaDeviceSynchronize();

  // Timing: Texture
  cudaEvent_t startTex, stopTex;
  cudaEventCreate(&startTex);
  cudaEventCreate(&stopTex);
  cudaEventRecord(startTex, 0);
  advectTextureKernel<<<blocks, threads>>>(texObj, d_output, WIDTH, HEIGHT);
  cudaEventRecord(stopTex, 0);
  cudaEventSynchronize(stopTex);
  float texTime = 0.0f;
  cudaEventElapsedTime(&texTime, startTex, stopTex);

  // Timing: Surface
  cudaEvent_t startSurf, stopSurf;
  cudaEventCreate(&startSurf);
  cudaEventCreate(&stopSurf);
  cudaEventRecord(startSurf, 0);
  advectSurfaceKernel<<<blocks, threads>>>(surfObj, WIDTH, HEIGHT);
  cudaEventRecord(stopSurf, 0);
  cudaEventSynchronize(stopSurf);
  float surfTime = 0.0f;
  cudaEventElapsedTime(&surfTime, startSurf, stopSurf);

  // Copy result back to host
  float *h_output = new float[WIDTH * HEIGHT];
  checkCuda(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost),
            "cudaMemcpy output");

  // Print results
  std::cout << "Texture + Global Memory Kernel Time: " << texTime << " ms"
            << std::endl;
  std::cout << "Surface Memory Kernel Time: " << surfTime << " ms" << std::endl;

  // Cleanup
  cudaDestroyTextureObject(texObj);
  cudaDestroySurfaceObject(surfObj);
  cudaFreeArray(cuArray);
  cudaFree(d_output);
  delete[] h_input;
  delete[] h_output;

  cudaEventDestroy(startTex);
  cudaEventDestroy(stopTex);
  cudaEventDestroy(startSurf);
  cudaEventDestroy(stopSurf);

  return 0;
}

// host_code_timed.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)

__constant__ float d_mask[MASK_WIDTH * MASK_WIDTH];

// Global memory kernel
__global__ void gaussianBlurGlobal(const float *input, float *output, int width,
                                   int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  if (x < width && y < height) {
    for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
      for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
        int xi = min(max(x + i, 0), width - 1);
        int yj = min(max(y + j, 0), height - 1);
        sum += input[yj * width + xi] *
               d_mask[(j + MASK_RADIUS) * MASK_WIDTH + (i + MASK_RADIUS)];
      }
    }
    output[y * width + x] = sum;
  }
}

// Texture memory kernel
__global__ void gaussianBlurTexture(cudaTextureObject_t texObj, float *output,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  if (x < width && y < height) {
    for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
      for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
        int xi = min(max(x + i, 0), width - 1);
        int yj = min(max(y + j, 0), height - 1);
        sum += tex2D<float>(texObj, xi + 0.5f, yj + 0.5f) *
               d_mask[(j + MASK_RADIUS) * MASK_WIDTH + (i + MASK_RADIUS)];
      }
    }
    output[y * width + x] = sum;
  }
}

// Load Gaussian mask
void loadGaussianMask() {
  float h_mask[MASK_WIDTH * MASK_WIDTH] = {1,  4, 6,  4,  1,  4, 16, 24, 16,
                                           4,  6, 24, 36, 24, 6, 4,  16, 24,
                                           16, 4, 1,  4,  6,  4, 1};
  float sum = 0.0f;
  for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
    sum += h_mask[i];
  for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
    h_mask[i] /= sum;
  cudaMemcpyToSymbol(d_mask, h_mask, sizeof(float) * MASK_WIDTH * MASK_WIDTH);
}

int main() {
  size_t size = WIDTH * HEIGHT * sizeof(float);
  float *h_input = (float *)malloc(size);
  float *h_output = (float *)malloc(size);

  for (int i = 0; i < WIDTH * HEIGHT; i++)
    h_input[i] = rand() % 256;

  float *d_input, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  loadGaussianMask();

  dim3 block(2024, 2024);
  dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

  // =======================
  // Global Memory Timing
  // =======================
  cudaEvent_t startGlobal, stopGlobal;
  cudaEventCreate(&startGlobal);
  cudaEventCreate(&stopGlobal);

  cudaEventRecord(startGlobal);
  gaussianBlurGlobal<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
  cudaEventRecord(stopGlobal);
  cudaEventSynchronize(stopGlobal);

  float timeGlobal;
  cudaEventElapsedTime(&timeGlobal, startGlobal, stopGlobal);

  // =======================
  // Texture Memory Setup
  // =======================
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT);
  cudaMemcpy2DToArray(cuArray, 0, 0, h_input, WIDTH * sizeof(float),
                      WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);

  struct cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // =======================
  // Texture Memory Timing
  // =======================
  cudaEvent_t startTex, stopTex;
  cudaEventCreate(&startTex);
  cudaEventCreate(&stopTex);

  cudaEventRecord(startTex);
  gaussianBlurTexture<<<grid, block>>>(texObj, d_output, WIDTH, HEIGHT);
  cudaEventRecord(stopTex);
  cudaEventSynchronize(stopTex);

  float timeTex;
  cudaEventElapsedTime(&timeTex, startTex, stopTex);

  // =======================
  // Print Results
  // =======================
  printf("Execution Time (Global Memory): %.3f ms\n", timeGlobal);
  printf("Execution Time (Texture Memory): %.3f ms\n", timeTex);

  if (timeTex < timeGlobal) {
    printf("✅ Texture memory is faster by %.3f ms (%.2f%% improvement)\n",
           timeGlobal - timeTex, 100.0f * (timeGlobal - timeTex) / timeGlobal);
  } else {
    printf("❌ Texture memory is slower by %.3f ms (%.2f%% slowdown)\n",
           timeTex - timeGlobal, 100.0f * (timeTex - timeGlobal) / timeGlobal);
  }

  // Cleanup
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  cudaEventDestroy(startGlobal);
  cudaEventDestroy(stopGlobal);
  cudaEventDestroy(startTex);
  cudaEventDestroy(stopTex);

  return 0;
}

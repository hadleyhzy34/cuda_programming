#include <cmath> // For exp, M_PI
#include <cuda_runtime.h>
#include <iomanip> // For std::fixed, std::setprecision
#include <iostream>
#include <vector>

// Helper for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// --- Gaussian Blur Kernel ---
// For simplicity, using a fixed 3x3 Gaussian kernel
// You could make this more general (pass kernel as constant memory, larger
// size) Kernel weights (normalized, sum to ~1) - approximate 1 2 1 2 4 2 1 2 1
// Divided by 16
__constant__ float d_gaussianKernel[3][3] = {
    {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
    {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
    {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}};

__global__ void gaussianBlurKernel(cudaTextureObject_t inputTex,
                                   float *outputImage, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float sum = 0.0f;

    // Iterate over the 3x3 neighborhood
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        // tex2D fetches a float.
        // Texture addressing mode (set during texture object creation) will
        // handle boundaries. We are using unnormalized coordinates here.
        float pixel_val =
            tex2D<float>(inputTex, static_cast<float>(x + kx) + 0.5f,
                         static_cast<float>(y + ky) + 0.5f);
        sum += pixel_val * d_gaussianKernel[ky + 1][kx + 1];
      }
    }
    outputImage[y * width + x] = sum;
  }
}

// --- Host-side Logic ---
void generateGrayscaleImage(float *image, int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Simple pattern for testing
      if ((x / 32) % 2 == (y / 32) % 2) {
        image[y * width + x] = 0.8f; // Light gray
      } else {
        image[y * width + x] = 0.2f; // Dark gray
      }
      // Add a bright spot
      if (x > width / 2 - 10 && x < width / 2 + 10 && y > height / 2 - 10 &&
          y < height / 2 + 10) {
        image[y * width + x] = 1.0f; // White
      }
    }
  }
}

void printImageSample(const float *image, int width, int height,
                      const std::string &title) {
  std::cout << "\n--- " << title << " (Sample 8x8 from top-left) ---\n";
  std::cout << std::fixed << std::setprecision(3);
  for (int y = 0; y < std::min(8, height); ++y) {
    for (int x = 0; x < std::min(8, width); ++x) {
      std::cout << image[y * width + x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "-------------------------------------------\n";
}

int main() {
  int width = 256;
  int height = 256;
  size_t imageSizeBytes = width * height * sizeof(float);

  // 1. Host memory for input and output
  float *h_inputImage = new float[width * height];
  float *h_outputImage = new float[width * height];

  generateGrayscaleImage(h_inputImage, width, height);
  printImageSample(h_inputImage, width, height, "Original Host Image");

  // 2. Device memory for output (linear global memory)
  float *d_outputImage;
  CUDA_CHECK(cudaMalloc(&d_outputImage, imageSizeBytes));

  // 3. Create CUDA Array for input
  cudaArray *cuArray_input;
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<float>(); // 32-bit float, 1 channel
  CUDA_CHECK(cudaMallocArray(&cuArray_input, &channelDesc, width, height,
                             cudaArrayDefault));

  // 4. Copy host input image to CUDA Array
  CUDA_CHECK(
      cudaMemcpy2DToArray(cuArray_input,         // dst array
                          0, 0,                  // dst x,y offset in array
                          h_inputImage,          // src host pointer
                          width * sizeof(float), // src pitch (bytes per row)
                          width * sizeof(float), // width of copy in bytes
                          height,                // height of copy
                          cudaMemcpyHostToDevice));

  // 5. Create Texture Object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray_input;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp x-coordinates
  texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp y-coordinates
  texDesc.filterMode = cudaFilterModePoint; // Nearest neighbor (Point sampling)
                                            // Could use cudaFilterModeLinear if
                                            // we wanted interpolation
  texDesc.readMode =
      cudaReadModeElementType; // Read elements as their native type (float)
  texDesc.normalizedCoords =
      0; // Use unnormalized pixel coordinates (0 to width-1, 0 to height-1)

  cudaTextureObject_t inputTexObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&inputTexObj, &resDesc, &texDesc, NULL));

  // 6. Kernel launch configuration
  dim3 blockSize(16, 16); // 256 threads per block
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  std::cout << "Launching Gaussian Blur kernel with grid: (" << gridSize.x
            << "," << gridSize.y << "," << gridSize.z << ") and block: ("
            << blockSize.x << "," << blockSize.y << "," << blockSize.z << ")"
            << std::endl;

  // 7. Launch kernel
  gaussianBlurKernel<<<gridSize, blockSize>>>(inputTexObj, d_outputImage, width,
                                              height);
  CUDA_CHECK(cudaGetLastError());      // Check for kernel launch errors
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete

  // 8. Copy output from device to host
  CUDA_CHECK(cudaMemcpy(h_outputImage, d_outputImage, imageSizeBytes,
                        cudaMemcpyDeviceToHost));
  printImageSample(h_outputImage, width, height, "Blurred Host Image");

  // 9. Cleanup
  CUDA_CHECK(cudaDestroyTextureObject(inputTexObj));
  CUDA_CHECK(cudaFreeArray(cuArray_input));
  CUDA_CHECK(cudaFree(d_outputImage));
  delete[] h_inputImage;
  delete[] h_outputImage;

  std::cout << "Gaussian blur example finished successfully." << std::endl;
  return 0;
}

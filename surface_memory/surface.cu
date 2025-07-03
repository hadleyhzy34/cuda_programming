#include <chrono>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <surface_functions.h>
#include <vector>

class SurfaceMemoryManager {
private:
  cudaArray_t cuda_array;
  cudaSurfaceObject_t surface_object;
  cudaTextureObject_t texture_object; // For comparison

public:
  // Initialize 2D surface memory
  void initializeSurface2D(int width, int height) {
    cudaError_t err;
    // 1. Create CUDA array descriptor
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

    // 2. Allocate CUDA array and future bind surface
    cudaMallocArray(&cuda_array, &channel_desc, width, height,
                    cudaArraySurfaceLoadStore);

    // Create surface objects
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;

    surface_object = 0;
    res_desc.res.array.array = cuda_array;
    err = cudaCreateSurfaceObject(&surface_object, &res_desc);
    if (err != cudaSuccess) {
      std::cerr << "cudaCreateSurfaceObject failed for input: "
                << cudaGetErrorString(err) << "\n";
      cudaFreeArray(cuda_array);
      return;
    }

    // 3. Create texture object
    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_array;

    // 4. Optional: Create texture object for read operations
    cudaTextureDesc texture_desc = {};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    cudaCreateTextureObject(&texture_object, &resource_desc, &texture_desc,
                            nullptr);
  }

  // Initialize 3D surface memory
  void initializeSurface3D(int width, int height, int depth) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

    cudaExtent extent = make_cudaExtent(width, height, depth);
    cudaMalloc3DArray(&cuda_array, &channel_desc, extent,
                      cudaArraySurfaceLoadStore);

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_array;

    cudaCreateSurfaceObject(&surface_object, &resource_desc);
  }

  cudaSurfaceObject_t getSurface() { return surface_object; }
  cudaTextureObject_t getTexture() { return texture_object; }
  cudaArray_t getArray() { return cuda_array; }

  ~SurfaceMemoryManager() {
    cudaDestroySurfaceObject(surface_object);
    cudaDestroyTextureObject(texture_object);
    cudaFreeArray(cuda_array);
  }
};

// ============================================================================
// SURFACE MEMORY ACCESS PATTERNS
// ============================================================================

// Basic surface read/write operations
__global__ void surfaceBasicOperations(cudaSurfaceObject_t surface, int width,
                                       int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // READ from surface
    float value;
    surf2Dread(&value, surface, x * sizeof(float), y);

    // WRITE to surface
    float new_value = value * 2.0f + 1.0f;
    surf2Dwrite(new_value, surface, x * sizeof(float), y);
  }
}

// Advanced: Gaussian blur using surface memory
__global__ void gaussianBlurSurface(cudaSurfaceObject_t input_surface,
                                    cudaSurfaceObject_t output_surface,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // printf("thread execution on %d,%d\n", x, y);

  // 5x5 Gaussian kernel
  const float kernel[5][5] = {
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};

  float sum = 0.0f;

  // Convolution with automatic boundary handling
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      // Surface not automatically handles boundary conditions
      if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
        float pixel;
        surf2Dread(&pixel, input_surface, (x + dx) * sizeof(float), y + dy);
        sum += pixel * kernel[dy + 2][dx + 2];
      }
    }
  }

  surf2Dwrite(sum, output_surface, x * sizeof(float), y);
}

// ============================================================================
// PERFORMANCE COMPARISON: GLOBAL vs TEXTURE vs SURFACE
// ============================================================================

// Global memory version (baseline)
__global__ void gaussianBlurGlobal(float *input, float *output, int width,
                                   int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const float kernel[5][5] = {
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};

  float sum = 0.0f;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int nx = x + dx;
      int ny = y + dy;

      // Manual boundary checking
      nx = max(0, min(width - 1, nx));
      ny = max(0, min(height - 1, ny));

      sum += input[ny * width + nx] * kernel[dy + 2][dx + 2];
    }
  }

  output[y * width + x] = sum;
}

// Texture memory version (read-only)
__global__ void gaussianBlurTexture(cudaTextureObject_t input_texture,
                                    float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const float kernel[5][5] = {
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f},
      {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f},
      {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};

  float sum = 0.0f;

  // printf("handling thread: %d, %d\n", x, y);

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      // Hardware interpolation and boundary handling
      float pixel = tex2D<float>(input_texture, x + dx + 0.5f, y + dy + 0.5f);
      sum += pixel * kernel[dy + 2][dx + 2];
    }
  }

  output[y * width + x] = sum;
}

// ============================================================================
// BEST PRACTICES AND OPTIMIZATION TECHNIQUES
// ============================================================================

// Best Practice 1: Use proper memory layout
class OptimizedSurfaceProcessor {
private:
  cudaArray_t cuda_array;
  cudaSurfaceObject_t surface;
  cudaStream_t stream;

public:
  void setupOptimizedSurface(int width, int height) {
    // Use proper channel format for your data
    cudaChannelFormatDesc desc;

    // For single precision float
    desc = cudaCreateChannelDesc<float>();

    // For multi-channel data (RGBA)
    // desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    // Allocate with optimal flags
    cudaMallocArray(&cuda_array, &desc, width, height,
                    cudaArraySurfaceLoadStore);

    // Create surface with resource descriptor
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaCreateSurfaceObject(&surface, &res_desc);

    // Create dedicated stream for surface operations
    cudaStreamCreate(&stream);
  }

  // Best Practice 2: Optimal kernel launch configuration
  void launchOptimizedKernel(int width, int height) {
    // Use 2D block dimensions that align with hardware
    dim3 block_size(16, 16); // 256 threads per block (optimal for most GPUs)
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // Launch with stream for better concurrency
    gaussianBlurSurface<<<grid_size, block_size, 0, stream>>>(surface, surface,
                                                              width, height);
  }
};

// Best Practice 3: Memory coalescing with surface memory
__global__ void coalescedSurfaceAccess(cudaSurfaceObject_t surface, int width,
                                       int height) {
  // Calculate global thread position
  int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Use shared memory to reduce surface accesses
  __shared__ float shared_data[18][18]; // 16x16 + 2 border

  int local_x = threadIdx.x + 1;
  int local_y = threadIdx.y + 1;

  // Load main data
  if (global_x < width && global_y < height) {
    surf2Dread(&shared_data[local_y][local_x], surface,
               global_x * sizeof(float), global_y);
  }

  // Load border data (edge threads load extra elements)
  if (threadIdx.x == 0 && global_x > 0) {
    surf2Dread(&shared_data[local_y][0], surface,
               (global_x - 1) * sizeof(float), global_y);
  }
  // ... similar for other borders

  __syncthreads();

  // Now process using shared memory instead of repeated surface reads
  if (global_x < width && global_y < height) {
    float result = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        result += shared_data[local_y + dy][local_x + dx] * 0.111f;
      }
    }

    surf2Dwrite(result, surface, global_x * sizeof(float), global_y);
  }
}

// ============================================================================
// PERFORMANCE BENCHMARKING
// ============================================================================

class SurfaceMemoryBenchmark {
public:
  void runBenchmarks(int width, int height, int iterations) {
    std::cout << "=== CUDA Surface Memory Benchmark ===" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Iterations: " << iterations << std::endl << std::endl;

    // Setup data
    std::vector<float> host_data(width * height);
    for (int i = 0; i < width * height; i++) {
      host_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Reset CUDA context to clear any prior errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Pre-benchmark CUDA error: " << cudaGetErrorString(err)
                << std::endl;
    }

    // Test Global Memory
    benchmarkGlobalMemory(host_data.data(), width, height, iterations);

    // Reset CUDA context
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after global memory benchmark: "
                << cudaGetErrorString(err) << std::endl;
    }

    // Test Surface Memory
    benchmarkSurfaceMemory(host_data.data(), width, height, iterations);

    // Reset CUDA context
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after surface memory benchmark: "
                << cudaGetErrorString(err) << std::endl;
    }

    // Test Texture Memory
    benchmarkTextureMemory(host_data.data(), width, height, iterations);

    // Final reset and check
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after texture memory benchmark: "
                << cudaGetErrorString(err) << std::endl;
    }
  }

private:
  void benchmarkGlobalMemory(float *data, int width, int height,
                             int iterations) {
    float *d_input, *d_output;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_input, width * height * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed for d_input: " << cudaGetErrorString(err)
                << std::endl;
      return;
    }
    err = cudaMalloc(&d_output, width * height * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed for d_output: " << cudaGetErrorString(err)
                << std::endl;
      cudaFree(d_input);
      return;
    }

    // Copy data to device
    err = cudaMemcpy(d_input, data, width * height * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpy failed for d_input: " << cudaGetErrorString(err)
                << std::endl;
      cudaFree(d_input);
      cudaFree(d_output);
      return;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
      gaussianBlurGlobal<<<grid, block>>>(d_input, d_output, width, height);
      std::swap(d_input, d_output);

      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "Surface memory kernel failed at iteration " << i
                  << cudaGetErrorString(err) << std::endl;
        break;
      }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Global memory kernel failed: " << cudaGetErrorString(err)
                << std::endl;
    }

    // Measure time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Global Memory: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void benchmarkSurfaceMemory(float *data, int width, int height,
                              int iterations) {
    SurfaceMemoryManager surface1, surface2;
    cudaError_t err;

    // Initialize surfaces
    surface1.initializeSurface2D(width, height);
    surface2.initializeSurface2D(width, height);

    // Copy data to surface
    err = cudaMemcpy2DToArray(surface1.getArray(), 0, 0, data,
                              width * sizeof(float), width * sizeof(float),
                              height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpy2DToArray failed for surface1: "
                << cudaGetErrorString(err) << std::endl;
      return;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run benchmark
    cudaEventRecord(start);
    // gaussianBlurSurface<<<grid, block>>>(surface1.getSurface(),
    //                                      surface2.getSurface(), width,
    //                                      height);
    // if (i % 2 == 0) {
    //   gaussianBlurSurface<<<grid, block>>>(
    //       surface1.getSurface(), surface2.getSurface(), width, height);
    // } else {
    //   gaussianBlurSurface<<<grid, block>>>(
    //       surface2.getSurface(), surface1.getSurface(), width, height);
    // }
    // cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   std::cerr << "Surface memory kernel failed at iteration " << ": "
    //             << cudaGetErrorString(err) << std::endl;
    //   return;
    // }
    for (int i = 0; i < iterations; i++) {
      gaussianBlurSurface<<<grid, block>>>(
          surface1.getSurface(), surface2.getSurface(), width, height);
      // gaussianBlurSurface<<<grid, block>>>(
      // surface1.getSurface(), surface2.getSurface(), width, height);
      // if (i % 2 == 0) {
      //   gaussianBlurSurface<<<grid, block>>>(
      //       surface1.getSurface(), surface2.getSurface(), width, height);
      // } else {
      //   gaussianBlurSurface<<<grid, block>>>(
      //       surface2.getSurface(), surface1.getSurface(), width, height);
      // }
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "Surface memory kernel failed at iteration " << i
                  << cudaGetErrorString(err) << std::endl;
        break;
      }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure time
    float milliseconds = 0;
    err = cudaEventElapsedTime(&milliseconds, start, stop);
    if (err == cudaSuccess) {
      std::cout << "Surface Memory: " << milliseconds << " ms" << std::endl;
    } else {
      std::cerr << "cudaEventElapsedTime failed for surface memory: "
                << cudaGetErrorString(err) << std::endl;
    }

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void benchmarkTextureMemory(float *data, int width, int height,
                              int iterations) {
    cudaError_t err;

    // 1. Device memory for output (linear global memory)
    float *d_outputImage;
    err = cudaMalloc(&d_outputImage, width * height * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed for d_outputImage: "
                << cudaGetErrorString(err) << std::endl;
      return;
    }

    // 2. Create CUDA Array for input
    cudaArray *cuArray_input;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    err = cudaMallocArray(&cuArray_input, &channelDesc, width, height,
                          cudaArrayDefault);
    if (err != cudaSuccess) {
      std::cerr << "cudaMallocArray failed: " << cudaGetErrorString(err)
                << std::endl;
      cudaFree(d_outputImage);
      return;
    }

    // 3. Copy host input image to CUDA Array
    err = cudaMemcpy2DToArray(cuArray_input, 0, 0, data, width * sizeof(float),
                              width * sizeof(float), height,
                              cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err)
                << std::endl;
      cudaFreeArray(cuArray_input);
      cudaFree(d_outputImage);
      return;
    }

    // 4. Create Texture Object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray_input;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t inputTexObj = 0;
    err = cudaCreateTextureObject(&inputTexObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
      std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(err)
                << std::endl;
      cudaFreeArray(cuArray_input);
      cudaFree(d_outputImage);
      return;
    }

    // 5. Set up kernel launch
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Run benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
      gaussianBlurTexture<<<grid, block>>>(inputTexObj, d_outputImage, width,
                                           height);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "Texture memory kernel failed at iteration " << i << ": "
                  << cudaGetErrorString(err) << std::endl;
        break;
      }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 7. Measure time
    float milliseconds = 0;
    err = cudaEventElapsedTime(&milliseconds, start, stop);
    if (err == cudaSuccess) {
      std::cout << "Texture Memory: " << milliseconds << " ms" << std::endl;
    } else {
      std::cerr << "cudaEventElapsedTime failed for texture memory: "
                << cudaGetErrorString(err) << std::endl;
    }

    // 8. Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDestroyTextureObject(inputTexObj);
    cudaFreeArray(cuArray_input);
    cudaFree(d_outputImage);
  }
};

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

int main() {
  // cudaArray *testArray;
  // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  // cudaError_t err =
  //     cudaMallocArray(&testArray, &channelDesc, 1024, 1024,
  //     cudaArrayDefault);
  // if (err != cudaSuccess) {
  //   std::cerr << "Test cudaMallocArray failed: " << cudaGetErrorString(err)
  //             << std::endl;
  // } else {
  //   std::cout << "Test cudaMallocArray succeeded!" << std::endl;
  //   cudaFreeArray(testArray);
  // }

  std::cout << "CUDA Surface Memory Deep Dive - CUDA 12.0+" << std::endl;
  std::cout << "================================================" << std::endl;

  // Check CUDA capability
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "Texture Memory: " << (prop.maxTexture2D[0]) << "x"
            << (prop.maxTexture2D[1]) << std::endl;
  std::cout << std::endl;

  // Run benchmarks
  SurfaceMemoryBenchmark benchmark;
  // benchmark.runBenchmarks(256, 256, 100);
  // benchmark.runBenchmarks(1024, 1024, 100);
  benchmark.runBenchmarks(500, 500, 100);

  std::cout << std::endl;
  std::cout << "=== WHEN TO USE SURFACE MEMORY ===" << std::endl;
  std::cout << "✅ 2D/3D grid computations with neighbor access" << std::endl;
  std::cout << "✅ Image processing and computer vision" << std::endl;
  std::cout << "✅ Scientific simulations (heat, fluid, cellular automata)"
            << std::endl;
  std::cout << "✅ Stencil computations" << std::endl;
  std::cout << "✅ When you need both read and write access to 2D data"
            << std::endl;
  std::cout << std::endl;
  std::cout << "❌ AVOID for linear 1D computations" << std::endl;
  std::cout << "❌ AVOID when atomic operations are critical" << std::endl;
  std::cout << "❌ AVOID for sparse data access patterns" << std::endl;

  return 0;
}

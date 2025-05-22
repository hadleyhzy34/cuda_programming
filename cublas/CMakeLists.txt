cmake_minimum_required(VERSION 3.10)
project(shared_memory LANGUAGES CXX CUDA)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Find CUDA
# find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# Find CUDA Toolkit (required for headers/libraries)
# find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(cublassaxpy src/cublasSaxpyTest.cu)

# Link CUDA libraries
target_link_libraries(cublassaxpy
  PRIVATE 
  # CUDA::cudart      # CUDA runtime
  CUDA::cublas      # Optional: If using cuBLAS
  # CUDA::cufft       # Optional: If using cuFFT
)

# Include directories (if needed)
target_include_directories(cublassaxpy PRIVATE include)

# # Link CUDA libraries
# target_link_libraries(transpose ${CUDA_LIBRARIES})

# Set CUDA architectures if needed
# set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_ARCHITECTURES "all")
# Enable CUDA architecture for your GPU (e.g., compute_86 for RTX 30xx)
set_target_properties(cublassaxpy PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Add CUDA specific flags
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)

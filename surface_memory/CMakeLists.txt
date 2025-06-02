cmake_minimum_required(VERSION 3.10)
project(surface_memory LANGUAGES CXX CUDA)
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
add_executable(surface surface.cu)
# add_executable(cublassaxpy src/cublasSaxpyTest.cu)

# Link CUDA libraries
target_link_libraries(surface
  PRIVATE 
  # CUDA::cudart      # CUDA runtime
  CUDA::cublas      # Optional: If using cuBLAS
  # CUDA::cufft       # Optional: If using cuFFT
)

# Include directories (if needed)
target_include_directories(surface PRIVATE include)

# Add executable
add_executable(surface-basic surface-basic.cu)
# add_executable(cublassaxpy src/cublasSaxpyTest.cu)

# Link CUDA libraries
target_link_libraries(surface-basic
  PRIVATE 
  # CUDA::cudart      # CUDA runtime
  CUDA::cublas      # Optional: If using cuBLAS
  # CUDA::cufft       # Optional: If using cuFFT
)

# Include directories (if needed)
target_include_directories(surface-basic PRIVATE include)

# Set CUDA architectures if needed
# set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_ARCHITECTURES "all")
# Enable CUDA architecture for your GPU (e.g., compute_86 for RTX 30xx)
set_target_properties(surface PROPERTIES
    CUDA_ARCHITECTURES "86"
)

cmake_minimum_required(VERSION 3.10)
project(constant_memory_and_events LANGUAGES CXX CUDA)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(dot_product main.cpp src/dot_product.h src/dot_product.cu)

# Link CUDA libraries
target_link_libraries(dot_product ${CUDA_LIBRARIES})

# Set CUDA architectures if needed
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add CUDA specific flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)

cmake_minimum_required(VERSION 3.10)
project(sobelcudademo LANGUAGES CXX CUDA)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (ensure OpenCV is installed and visible to CMake)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install it.")
endif()

# Configure CUDA (compute capability should match your GPU's architecture)
set(CMAKE_CUDA_ARCHITECTURES "75")  # e.g., 75 for Turing (RTX 20xx), 52 for Maxwell, etc.
set(CMAKE_CUDA_STANDARD 14)         # C++14 standard

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(main src/cudaArray.cu)

# Link OpenCV and CUDA libraries
target_link_libraries(main
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# Add CUDA specific flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)

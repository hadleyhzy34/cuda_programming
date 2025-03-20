cmake_minimum_required(VERSION 3.10)
project(sobelcudademo LANGUAGES CXX CUDA)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find opencv (ensure opencv is installed and visible to cmake)
find_package(CUDA REQUIRED)
# Find OpenCV package
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install it.")
endif()

# configure cuda (compute capability should match your gpu's architecture)
set(cmake_cuda_architectures "75")  # e.g., 75 for turing (rtx 20xx), 52 for maxwell, etc.
set(cmake_cuda_standard 14)         # c++14 standard

#Include OPenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# add executable
add_executable(sobel_cuda
  main.cpp
  src/constantSobelFilter.hpp
  src/constantSobelFilter.cu
  src/sobelFilter.hpp
  src/sobelFilter.cu
)

# link opencv and cuda libraries
target_link_libraries(sobel_cuda
  # private
  ${OpenCV_LIBS}
)

# # include opencv headers
# target_include_directories(sobel_cuda
#   private
#   ${OpenCV_INCLUDE_DIRS}
# )

set(CMAKE_CUDA_ARCHITECTURES "all")

# Add CUDA specific flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)

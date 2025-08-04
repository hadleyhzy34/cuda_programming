# --- 1. Set Minimum CMake Version and Project Info ---
# Modern CUDA support was introduced around 3.10, 3.18+ is recommended.
cmake_minimum_required(VERSION 3.18)

# Define the project name and enable C++ (CXX) and CUDA as languages.
# This is the key step to tell CMake to find and use nvcc.
project(CUDAMerge LANGUAGES CXX CUDA)

# --- 2. Configure C++ and CUDA Standards ---
# Set the C++ standard to C++11, which our code uses.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Also set the CUDA standard. This is important!
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")

# Set default build type to Debug if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# --- 3. Specify the GPU Architecture (Critical for your Tesla T4) ---
# This is the CMake equivalent of the nvcc flag:
# -gencode arch=compute_75,code=sm_75
#
# The value '75' corresponds to the Tesla T4's "Turing" architecture.
# CMake handles the "arch" and "code" parts for you.
set(CMAKE_CUDA_ARCHITECTURES 75)

# --- 4. Define the Executable ---
# Tell CMake to create an executable named 'merge_app' from 'merge.cu'
add_executable(main main.cu)

# --- 5. (Optional but good practice) Print Info ---
# Display the CUDA architectures being compiled for.
message(STATUS "Compiling for CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

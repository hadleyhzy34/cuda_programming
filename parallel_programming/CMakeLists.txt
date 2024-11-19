cmake_minimum_required(VERSION 3.8)
project(parallel_programming LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# add_library(utility src/utility.h src/utility.cu)
# target_compile_features(utility PUBLIC cxx_std_11)

# add_library(new src/new.cu)
# target_compile_features(new PUBLIC cxx_std_11)

add_executable(julia_set src/julia_set.cu)

add_executable(add src/add_vectors.cu)

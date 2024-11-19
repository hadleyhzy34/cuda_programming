cmake_minimum_required(VERSION 3.8)
project(basics LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(utility src/utility.h src/utility.cu)
target_compile_features(utility PUBLIC cxx_std_11)

# add_library(new src/new.cu)
# target_compile_features(new PUBLIC cxx_std_11)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} utility)

cmake_minimum_required(version 3.10)
project(shared_memory languages cxx cuda)
enable_language(cuda)

# set c++ standard
set(cmake_cxx_standard 17)
set(cmake_cxx_standard_required on)

# # find cuda
# find_package(cuda required)
find_package(cudatoolkit required)
# include_directories(${cuda_include_dirs})
# find cuda toolkit (required for headers/libraries)
# find_package(cudatoolkit required)

# add executable
add_executable(main main.cu)
# add_executable(cudastream main-stream.cu)
# add_executable(cuda-graph main-graph.cu)
# add_executable(cuda-benchmark main-benchmark.cu)
# add_executable(cuda-advanced main-advanced.cu)

# link cuda libraries
target_link_libraries(main
  private 
  # cuda::cudart      # cuda runtime
  # cuda::cublas      # optional: if using cublas
  # cuda::cufft       # optional: if using cufft
)

# include directories (if needed)
target_include_directories(main private include)
# target_include_directories(cudastream private include)
# target_include_directories(cuda-graph private include)
# target_include_directories(cuda-benchmark private include)
# target_include_directories(cuda-advanced private include)

# # link cuda libraries
# target_link_libraries(transpose ${cuda_libraries})

# set cuda architectures if needed
# set(cmake_cuda_architectures 75)
# set(cmake_cuda_architectures "all")
# enable cuda architecture for your gpu (e.g., compute_86 for rtx 30xx)
set_target_properties(main properties
    cuda_architectures "86"
)

# add cuda specific flags
# set(cuda_nvcc_flags ${cuda_nvcc_flags};-o2)

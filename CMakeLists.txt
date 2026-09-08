cmake_minimum_required(VERSION 3.18)
project(TensorizedNewtonSchulz LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(CUDAToolkit REQUIRED)

# 1. Point to the root utils directory instead of src/utils
add_subdirectory(utils)
add_subdirectory(src/common)
add_subdirectory(benchmarks)
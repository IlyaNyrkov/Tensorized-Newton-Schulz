add_library(cans_common STATIC
        nested_polynomial.cu
)

# Allow other targets to #include "nested_polynomial.cuh" by just targeting this directory
target_include_directories(cans_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link cuBLAS
target_link_libraries(cans_common PUBLIC CUDA::cublas)
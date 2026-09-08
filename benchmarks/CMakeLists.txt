add_executable(bench_01_nested_polynomial 01_nested_polynomial.cu)

# 2. Swap 'cans_utils' for the new 'matrix_utils' interface target
target_link_libraries(bench_01_nested_polynomial
        PRIVATE
        cans_common
        matrix_utils
        CUDA::cublas
)
# List of benchmark names (without .cu extension)
set(BENCHMARKS
        01_time_to_target
        02_fixed_budget
        03_peterson_stockeyer_ns
)

foreach(bench IN LISTS BENCHMARKS)
    add_executable(${bench} ${bench}.cu)
    target_link_libraries(${bench} PRIVATE
            cans_common
            cans_utils
    )
endforeach()
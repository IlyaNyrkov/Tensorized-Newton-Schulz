# Declare a header-only interface library
add_library(cans_common INTERFACE)

# Expose both the public API (include) and private headers (src/common)
# to anything that links against cans_common
target_include_directories(cans_common INTERFACE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Propagate the cuBLAS dependency to the executables
target_link_libraries(cans_common INTERFACE cublas)
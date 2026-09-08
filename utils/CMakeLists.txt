# ==========================================
# UTILITIES LIBRARY (Header-Only)
# ==========================================

# We define this as an INTERFACE library because template implementations
# Must reside in the header files. No separate compilation is required.
add_library(matrix_utils INTERFACE)

# Expose the current directory so that targets linking against matrix_utils
# can simply use: #include "matrix_gen.hpp"
target_include_directories(matrix_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
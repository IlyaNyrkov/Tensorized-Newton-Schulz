#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <vector>

#include "cans_lookup_table.cuh"

namespace cans {

// -----------------------------------------------------------------------------
// Optional Profiling Struct (RAII wrapper for CUDA events)
// -----------------------------------------------------------------------------
    struct CansTimings {
        float phase0_ms = 0.0f;  // Gelfand Normalization
        float phase1_ms = 0.0f;  // Nested Polynomials
        float phase2_ms = 0.0f;  // Newton-Schulz Iterations
        float total_ms = 0.0f;

        cudaEvent_t events[4];

        CansTimings() {
            for (int i = 0; i < 4; ++i) cudaEventCreate(&events[i]);
        }

        ~CansTimings() {
            for (int i = 0; i < 4; ++i) cudaEventDestroy(events[i]);
        }

        // Disable copy/assignment to prevent double-freeing the underlying CUDA
        // events
        CansTimings(const CansTimings&) = delete;
        CansTimings& operator=(const CansTimings&) = delete;
    };

// -----------------------------------------------------------------------------
// Memory Workspace
// -----------------------------------------------------------------------------
    template <typename T>
    class Workspace {
    public:
        std::size_t n;
        std::size_t p;
        cudaStream_t stream;
        cublasHandle_t handle;

        std::vector<std::vector<float>> phase1_polynomials;
        std::vector<std::array<float, 2>> phase2_iterations;

        T* d_X_next;
        T* d_M;
        T* d_temp1;
        T* d_temp2;
        T* d_scalar_ws;

        Workspace(std::size_t n_rows, std::size_t p_cols, float delta,
                  cudaStream_t execution_stream = nullptr)
                : n(n_rows), p(p_cols), stream(execution_stream) {
            cans::lookup_table::CansConfig config =
                    cans::lookup_table::get_config(delta);
            phase1_polynomials = config.phase1_polynomials;
            phase2_iterations = config.phase2_iterations;

            cublasCreate(&handle);
            cublasSetStream(handle, stream);

            // Use size_t for size computations to avoid overflow.
            // For very large allocations, consider adding overflow checks,
            // e.g. verifying that n * p <= SIZE_MAX / sizeof(T).
            const std::size_t x_next_bytes = n * p * sizeof(T);
            const std::size_t p_squared_bytes = p * p * sizeof(T);

            cudaMallocAsync(&d_X_next, x_next_bytes, stream);
            cudaMallocAsync(&d_M, p_squared_bytes, stream);
            cudaMallocAsync(&d_temp1, p_squared_bytes, stream);
            cudaMallocAsync(&d_temp2, p_squared_bytes, stream);
            cudaMallocAsync(&d_scalar_ws, 2 * sizeof(T), stream);
        }

        ~Workspace() {
            cudaFree(d_X_next);
            cudaFree(d_M);
            cudaFree(d_temp1);
            cudaFree(d_temp2);
            cudaFree(d_scalar_ws);
            cublasDestroy(handle);
        }

        Workspace(const Workspace&) = delete;
        Workspace& operator=(const Workspace&) = delete;
    };

}  // namespace cans
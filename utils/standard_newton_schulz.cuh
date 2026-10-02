#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

namespace cans_utils {
    namespace reference {

// -----------------------------------------------------------------------------
// Standard Frobenius Normalization
// Divides the matrix by its Frobenius norm to force singular values <= 1.0
// -----------------------------------------------------------------------------
        template <typename T>
        void apply_standard_normalization(
                cublasHandle_t handle,
                T* d_X,
                T* d_scalar_ws,
                int m,
                int n,
                cudaStream_t stream = nullptr
        ) {
            // 1. Compute ||X||_F
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            cublasSnrm2(handle, m * n, d_X, 1, d_scalar_ws);

            // 2. Fetch norm to host to compute the inverse scale
            // (A host sync is acceptable here since this is just a benchmark reference)
            T h_norm;
            cudaMemcpyAsync(&h_norm, d_scalar_ws, sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            T scale = (h_norm > 1e-12f) ? (1.0f / h_norm) : 1.0f;

            // 3. Scale X in-place
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            cublasSscal(handle, m * n, &scale, d_X, 1);
        }

// -----------------------------------------------------------------------------
// Standard Newton-Schulz Iteration (X_{k+1} = 1.5*X_k - 0.5*X_k*X_k^T*X_k)
// -----------------------------------------------------------------------------
        template <typename T>
        void apply_standard_ns_iterations(
                cublasHandle_t handle,
                T* d_X,
                T* d_X_next,
                T* d_M,
                int m,
                int n,
                int iterations,
                cudaStream_t stream = nullptr
        ) {
            T alpha_1 = 1.0f, beta_0 = 0.0f;
            T alpha_gemm = -0.5f, beta_gemm = 1.5f;

            T* current_X = d_X;
            T* next_X = d_X_next;

            for (int k = 0; k < iterations; ++k) {
                // M = X^T * X
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m,
                            &alpha_1, current_X, n, current_X, n, &beta_0, d_M, n);

                cudaMemcpyAsync(next_X, current_X, m * n * sizeof(T), cudaMemcpyDeviceToDevice, stream);

                // X_{k+1} = -0.5 * (X_k * M) + 1.5 * X_{k+1}
                // Row-major X_k * M maps to col-major M' * X_k'
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n,
                            &alpha_gemm, d_M, n, current_X, n, &beta_gemm, next_X, n);

                std::swap(current_X, next_X);
            }

            // Ensure final result lands in the original user pointer
            if (current_X != d_X) {
                cudaMemcpyAsync(d_X, current_X, m * n * sizeof(T), cudaMemcpyDeviceToDevice, stream);
            }
        }

    } // namespace reference
} // namespace cans_utils
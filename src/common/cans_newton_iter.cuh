#pragma once

#include <vector>
#include <array>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <utility> // For std::swap

namespace cans {
    void apply_cans_pure_blas(
            cublasHandle_t handle,
            float *d_X,
            int n,
            int p,
            const std::vector <std::array<float, 2>> &coeffs,
            cudaStream_t stream
    ) {
        float alpha_1 = 1.0f, beta_0 = 0.0f;

        // Allocate device workspace for the Gram matrix and pointer swapping
        float *d_M, *d_X_next;
        cudaMallocAsync(&d_M, p * p * sizeof(float), stream);
        cudaMallocAsync(&d_X_next, n * p * sizeof(float), stream);

        float *current_X = d_X;
        float *next_X = d_X_next;

        for (const auto& c : coeffs) {
            // M = X^T * X
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n,
                        &alpha_1, current_X, p, current_X, p, &beta_0, d_M, p);

            cudaMemcpyAsync(next_X, current_X, n * p * sizeof(float), cudaMemcpyDeviceToDevice, stream);

            // X_{j+1} = -c3 * (X_j * M) + c1 * X_{j+1}
            float alpha_gemm = -c[1];
            float beta_gemm = c[0];
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, n, p,
                        &alpha_gemm, d_M, p, current_X, p, &beta_gemm, next_X, p);

            std::swap(current_X, next_X);
        }

        // Ensure the final state lands in the original user-provided pointer
        if (current_X != d_X) {
            cudaMemcpyAsync(d_X, current_X, n * p * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        cudaFreeAsync(d_X_next, stream);
        cudaFreeAsync(d_M, stream);
    }
}
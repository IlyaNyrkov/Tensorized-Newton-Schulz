#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace cans {
    __global__ void scale_add_diag_kernel(const float *M, float *out, float alpha,
                                          float beta, int p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < p * p) {
            int row = idx % p;
            int col = idx / p;
            float val = alpha * M[idx];
            if (row == col) {
                val += beta;
            }
            out[idx] = val;
        }
    }

    void launch_scale_add_diag(const float *d_in, float *d_out, float alpha,
                               float beta, int p, cudaStream_t stream) {
        int threads = 256;
        int blocks = (p * p + threads - 1) / threads;
        scale_add_diag_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, alpha,
                                                              beta, p);
    }

    void apply_degree_3_poly(cublasHandle_t handle, float *d_X, float *d_X_out,
                             float *d_M, float *d_temp1, const float *coeffs, int n,
                             int p, cudaStream_t stream) {
        float alpha_1 = 1.0f, beta_0 = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X,
                    n, &beta_0, d_M, p);
        launch_scale_add_diag(d_M, d_temp1, coeffs[1], coeffs[0], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n,
                    d_temp1, p, &beta_0, d_X_out, n);
        cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream);
    }

    void apply_degree_5_poly(cublasHandle_t handle, float *d_X, float *d_X_out,
                             float *d_M, float *d_temp1, float *d_temp2,
                             const float *coeffs, int n, int p,
                             cudaStream_t stream) {
        float alpha_1 = 1.0f, beta_0 = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X,
                    n, &beta_0, d_M, p);
        launch_scale_add_diag(d_M, d_temp1, coeffs[2], coeffs[1], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p,
                    d_temp1, p, &beta_0, d_temp2, p);
        launch_scale_add_diag(d_temp2, d_temp2, 1.0f, coeffs[0], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n,
                    d_temp2, p, &beta_0, d_X_out, n);
        cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream);
    }

    void apply_degree_7_poly(cublasHandle_t handle, float *d_X, float *d_X_out,
                             float *d_M, float *d_temp1, float *d_temp2,
                             const float *coeffs, int n, int p,
                             cudaStream_t stream) {
        float alpha_1 = 1.0f, beta_0 = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X,
                    n, &beta_0, d_M, p);
        launch_scale_add_diag(d_M, d_temp1, coeffs[3], coeffs[2], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p,
                    d_temp1, p, &beta_0, d_temp2, p);
        launch_scale_add_diag(d_temp2, d_temp2, 1.0f, coeffs[1], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p,
                    d_temp2, p, &beta_0, d_temp1, p);
        launch_scale_add_diag(d_temp1, d_temp1, 1.0f, coeffs[0], p, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n,
                    d_temp1, p, &beta_0, d_X_out, n);
        cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream);
    }

    void apply_nested_polynomial(cublasHandle_t handle, float *d_X, int n, int p,
                                 const std::vector<std::vector<float>> &polynomials,
                                 cudaStream_t stream) {
        size_t x_out_elems = n * p;
        size_t square_elems = p * p;

        float *d_workspace;
        size_t total_bytes = (x_out_elems + 3 * square_elems) * sizeof(float);
        cudaMallocAsync(&d_workspace, total_bytes, stream);

        float *d_X_out = d_workspace;
        float *d_M = d_workspace + x_out_elems;
        float *d_temp1 = d_M + square_elems;
        float *d_temp2 = d_temp1 + square_elems;

        for (const auto &coeffs : polynomials) {
            switch (coeffs.size()) {
                case 2:
                    apply_degree_3_poly(handle, d_X, d_X_out, d_M, d_temp1, coeffs.data(),
                                        n, p, stream);
                    break;
                case 3:
                    apply_degree_5_poly(handle, d_X, d_X_out, d_M, d_temp1, d_temp2,
                                        coeffs.data(), n, p, stream);
                    break;
                case 4:
                    apply_degree_7_poly(handle, d_X, d_X_out, d_M, d_temp1, d_temp2,
                                        coeffs.data(), n, p, stream);
                    break;
                default:
                    cudaFreeAsync(d_workspace, stream);
                    throw std::runtime_error("Unsupported polynomial coefficient count.");
            }
        }
        cudaFreeAsync(d_workspace, stream);
    }
}  // namespace cans
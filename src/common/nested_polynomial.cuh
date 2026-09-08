#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Launches the custom fused element-wise scaling and diagonal addition kernel
void launch_scale_add_diag(const float* d_in, float* d_out, float alpha, float beta, int p, cudaStream_t stream = nullptr);

// Specialized unrolled polynomial executors
void apply_degree_3_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, const float* coeffs, int n, int p, cudaStream_t stream);
void apply_degree_5_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, float* d_temp2, const float* coeffs, int n, int p, cudaStream_t stream);
void apply_degree_7_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, float* d_temp2, const float* coeffs, int n, int p, cudaStream_t stream);

// Main dispatcher function
void apply_nested_polynomial(cublasHandle_t handle, float* d_X, int n, int p, const std::vector<std::vector<float>>& polynomials, cudaStream_t stream = nullptr);
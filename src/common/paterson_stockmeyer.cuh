#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <vector>

// Macro for CUDA error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for cuBLAS error checking
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ---------------------------------------------------------
// CUDA Kernels for Element-wise Operations
// ---------------------------------------------------------

// Kernel to form a Paterson-Stockmeyer block B_i
// B_i = c_0 I + c_1 M + c_2 M^2 + ... + c_{s-1} M^{s-1}
__global__ void build_ps_block_kernel(float* B, const float* M, const float* M_powers,
                                      const float* coeffs, int s_actual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;

    if (idx < total_elements) {
        int row = idx % n;
        int col = idx / n;

        // Start with c_0 I
        float val = (row == col) ? coeffs[0] : 0.0f;

        // Add c_1 M
        if (s_actual > 1) {
            val += coeffs[1] * M[idx];
        }

        // Add c_k M^k for k >= 2
        for (int k = 2; k < s_actual; ++k) {
            // M_powers stores M^2, M^3... consecutively
            int power_offset = (k - 2) * total_elements;
            val += coeffs[k] * M_powers[power_offset + idx];
        }
        B[idx] = val;
    }
}

// Kernel for standard Horner addition: Res = Res + c I
__global__ void add_scalar_identity_kernel(float* matrix, float c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        if (row == col) {
            matrix[idx] += c;
        }
    }
}

// ---------------------------------------------------------
// Paterson-Stockmeyer Workspace
// ---------------------------------------------------------

struct psWorkspace {
    uint n;
    uint degree;   // Degree of P(M)
    uint s;        // Block size (~sqrt(degree))
    uint r;        // Number of blocks (degree / s)

    float* M;           // Gram matrix M (n x n)
    float* M_powers;    // M^2 to M^s ((s-1) * n * n elements)
    float* blocks;      // B_0 to B_r ((r+1) * n * n elements)
    float* P_M;         // Final evaluated polynomial (n x n)
    float* temp_gemm;   // Temporary buffer for GEMM operations (n x n)

    psWorkspace(uint polyDegree, uint matrix_N) : degree(polyDegree), n(matrix_N) {
        // 1. Calculate PS block parameters
        // For degree d, optimal s is ceil(sqrt(d))
        s = std::ceil(std::sqrt(degree + 1));
        if (s < 2 && degree >= 1) s = 2;
        r = degree / s;

        uint n2 = n * n;

        // 2. Allocate GPU memory
        CHECK_CUDA(cudaMalloc(&M, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&P_M, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&temp_gemm, n2 * sizeof(float)));

        if (s > 2) {
            CHECK_CUDA(cudaMalloc(&M_powers, (s - 2) * n2 * sizeof(float)));
        } else {
            M_powers = nullptr; // M^2 is not needed if s <= 2
        }

        CHECK_CUDA(cudaMalloc(&blocks, (r + 1) * n2 * sizeof(float)));
    }

    ~psWorkspace() {
        cudaFree(M);
        cudaFree(P_M);
        cudaFree(temp_gemm);
        if (M_powers) cudaFree(M_powers);
        cudaFree(blocks);
    }
};

// ---------------------------------------------------------
// Polynomial Evaluations
// ---------------------------------------------------------

// Evaluates P(M) using Paterson-Stockmeyer and stores in ws.P_M
void polynomialPS(cublasHandle_t handle, psWorkspace& ws, float* d_coeffs) {
    uint n = ws.n;
    uint n2 = n * n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    int threads = 256;

    // 1. Calculate Matrix Powers M^2, M^3 ... M^{s}
    // Note: We already have M. M_powers[0] will be M^2.
    if (ws.s >= 2) {
        // M^2 = M * M
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, ws.M, n, ws.M, n, &beta, ws.M_powers, n));
        // M^k = M * M^{k-1}
        for (uint k = 3; k <= ws.s; ++k) {
            float* prev_power = ws.M_powers + (k - 3) * n2;
            float* next_power = ws.M_powers + (k - 2) * n2;
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &alpha, ws.M, n, prev_power, n, &beta, next_power, n));
        }
    }

    // Pointer to M^s (used in Horner's evaluation)
    float* M_s = (ws.s == 1) ? ws.M : (ws.M_powers + (ws.s - 2) * n2);


    int blocks_grid = (n2 + threads - 1) / threads;

    // 2. Calculate blocks B_i
    for (uint i = 0; i <= ws.r; ++i) {
        uint s_actual = std::min(ws.s, ws.degree - i * ws.s + 1);
        float* current_block = ws.blocks + i * n2;
        float* current_coeffs = d_coeffs + i * ws.s;

        build_ps_block_kernel<<<blocks_grid, threads>>>(
                current_block, ws.M, ws.M_powers, current_coeffs, s_actual, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. Evaluate using Blocked Horner's Method: B_0 + M^s (B_1 + M^s B_2 ...)
    // Initialize Res with highest block B_r
    CHECK_CUDA(cudaMemcpy(ws.P_M, ws.blocks + ws.r * n2, n2 * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int i = ws.r - 1; i >= 0; --i) {
        // temp = Res * M^s
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, ws.P_M, n, M_s, n, &beta, ws.temp_gemm, n));

        // Res = temp + B_i
        // (Achieved via cublasSgeam for matrix addition)
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                                 &alpha, ws.temp_gemm, n, &alpha, ws.blocks + i * n2, n,
                                 ws.P_M, n));
    }
}

// ---------------------------------------------------------
// Main Newton-Schulz Execution Functions
// ---------------------------------------------------------

void calculateNS_PS(cublasHandle_t handle, psWorkspace& ws, float* d_X, float* d_Res, float* d_coeffs, uint m) {
    uint n = ws.n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 1. GEMM M = X^T * X
    // Layout: X is (m x n) column-major. X^T is (n x m).
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &alpha, d_X, m, d_X, m, &beta, ws.M, n));

    // 2. Calculate P(M) using Paterson-Stockmeyer
    polynomialPS(handle, ws, d_coeffs);

    // 3. GEMM Res = X * P(M)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &alpha, d_X, m, ws.P_M, n, &beta, d_Res, m));
}

// Naive Baseline: Standard Factorized Horner's Method on the Gram Matrix
void calculateNS_Naive(cublasHandle_t handle, float* d_X, float* d_Res, float* d_coeffs, uint m, uint n, uint degree) {
    uint n2 = n * n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    int threads = 256;

    // Allocate internal buffers for the naive factorized approach
    float *d_M, *d_PM, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_M, n2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_PM, n2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, n2 * sizeof(float)));

    // 1. GEMM M = X^T * X
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &alpha, d_X, m, d_X, m, &beta, d_M, n));

    // 2. Standard Horner: P(M) = c_0 I + M(c_1 I + M(c_2 I + ...))

    // Copy highest coefficient to Host to initialize P(M) = c_d I
    float c_highest;
    CHECK_CUDA(cudaMemcpy(&c_highest, d_coeffs + degree, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(d_PM, 0, n2 * sizeof(float)));

    int blocks_grid = (n2 + threads - 1) / threads;
    add_scalar_identity_kernel<<<blocks_grid, threads>>>(d_PM, c_highest, n);

    // Evaluate inwards
    for (int i = degree - 1; i >= 0; --i) {
        // temp = P_M * M
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, d_PM, n, d_M, n, &beta, d_temp, n));

        // P_M = temp
        CHECK_CUDA(cudaMemcpy(d_PM, d_temp, n2 * sizeof(float), cudaMemcpyDeviceToDevice));

        // P_M = P_M + c_i I
        float c_current;
        CHECK_CUDA(cudaMemcpy(&c_current, d_coeffs + i, sizeof(float), cudaMemcpyDeviceToHost));
        add_scalar_identity_kernel<<<blocks_grid, threads>>>(d_PM, c_current, n);
    }

    // 3. Res = X * P(M)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &alpha, d_X, m, d_PM, n, &beta, d_Res, m));

    cudaFree(d_M);
    cudaFree(d_PM);
    cudaFree(d_temp);
}
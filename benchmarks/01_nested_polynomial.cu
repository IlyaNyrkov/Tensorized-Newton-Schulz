#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "nested_polynomial.cuh"
#include "cans_constants.hpp"
#include "matrix_gen.hpp"
#include "matrix_compare.hpp"

// -----------------------------------------------------------------------------
// Naive Baseline Implementation
// Uses standard cuBLAS calls exclusively (cublasSgeam, cublasSgemm)
// Requires an explicit, dense identity matrix on the device.
// -----------------------------------------------------------------------------
void apply_nested_polynomial_naive(cublasHandle_t handle, float* d_X, int n, int p,
                                   const std::vector<std::vector<float>>& polynomials,
                                   cudaStream_t stream = nullptr) {
    float one = 1.0f, zero = 0.0f;

    // 1. Allocate workspace: X_out, M, temp1, temp2, plus an explicit Identity matrix
    size_t x_elems = n * p;
    size_t p2_elems = p * p;
    float *d_workspace, *d_Eye;

    cudaMallocAsync(&d_workspace, (x_elems + 3 * p2_elems) * sizeof(float), stream);
    cudaMallocAsync(&d_Eye, p2_elems * sizeof(float), stream);

    // Initialize physical identity matrix on device
    std::vector<float> h_Eye(p2_elems, 0.0f);
    for (int i = 0; i < p; ++i) h_Eye[i * p + i] = 1.0f;
    cudaMemcpyAsync(d_Eye, h_Eye.data(), p2_elems * sizeof(float), cudaMemcpyHostToDevice, stream);

    float* d_X_out = d_workspace;
    float* d_M     = d_workspace + x_elems;
    float* d_temp1 = d_M + p2_elems;
    float* d_temp2 = d_temp1 + p2_elems;

    for (const auto& coeffs : polynomials) {
        // M = X^T * X
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &one, d_X, n, d_X, n, &zero, d_M, p);

        if (coeffs.size() == 2) {
            // Degree 3: Q = c3 * M + c1 * I using cublasSgeam
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &coeffs[1], d_M, p, &coeffs[0], d_Eye, p, d_temp1, p);
            // X_out = X * Q
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &one, d_X, n, d_temp1, p, &zero, d_X_out, n);
        }
        else if (coeffs.size() == 3) {
            // Degree 5: K = c5 * M + c3 * I
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &coeffs[2], d_M, p, &coeffs[1], d_Eye, p, d_temp1, p);
            // temp2 = M * K
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &one, d_M, p, d_temp1, p, &zero, d_temp2, p);
            // Q = 1.0 * temp2 + c1 * I
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &one, d_temp2, p, &coeffs[0], d_Eye, p, d_temp1, p);
            // X_out = X * Q
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &one, d_X, n, d_temp1, p, &zero, d_X_out, n);
        }
        else if (coeffs.size() == 4) {
            // Degree 7: L = c7 * M + c5 * I
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &coeffs[3], d_M, p, &coeffs[2], d_Eye, p, d_temp1, p);
            // temp2 = M * L
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &one, d_M, p, d_temp1, p, &zero, d_temp2, p);
            // K = 1.0 * temp2 + c3 * I
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &one, d_temp2, p, &coeffs[1], d_Eye, p, d_temp1, p);
            // temp2 = M * K
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &one, d_M, p, d_temp1, p, &zero, d_temp2, p);
            // Q = 1.0 * temp2 + c1 * I
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, &one, d_temp2, p, &coeffs[0], d_Eye, p, d_temp1, p);
            // X_out = X * Q
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &one, d_X, n, d_temp1, p, &zero, d_X_out, n);
        }

        cudaMemcpyAsync(d_X, d_X_out, x_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaFreeAsync(d_Eye, stream);
    cudaFreeAsync(d_workspace, stream);
}

// -----------------------------------------------------------------------------
// Benchmark Runner
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int rows = 4096;
    const int cols = 2048;
    const int warmup_runs = 3;
    const int benchmark_runs = 10;

    std::cout << "========================================================\n";
    std::cout << " Benchmark: Phase 1 Nested Polynomial Application\n";
    std::cout << " Matrix Dimensions: " << rows << " x " << cols << "\n";
    std::cout << " Warmup: " << warmup_runs << " | Measured Iterations: " << benchmark_runs << "\n";
    std::cout << "========================================================\n\n";

    // Initialize cuBLAS & Stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // Stage coefficients for Delta = 0.00443 (Sequence: [3, 7, 7])
    std::vector<std::vector<float>> polynomials = {
            {4.55741422f, -4.35240249f},
            {3.63058296f, -4.96518992f, 2.58740030f, -0.42160228f},
            {2.30216330f, -2.42692774f, 1.44332413f, -0.32270912f}
    };

    // Host matrices
    std::vector<float> h_X_init = matrix_utils::generation::generate_matrix<float>(rows, cols, 0.5f, 1337);
    std::vector<float> h_X_naive_result(rows * cols);
    std::vector<float> h_X_opt_result(rows * cols);

    // Device allocations
    float *d_X_naive, *d_X_opt;
    size_t matrix_bytes = rows * cols * sizeof(float);
    cudaMalloc(&d_X_naive, matrix_bytes);
    cudaMalloc(&d_X_opt, matrix_bytes);

    // Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------------------------------------------------
    // 1. Benchmark Naive Implementation
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_runs; ++i) {
        cudaMemcpyAsync(d_X_naive, h_X_init.data(), matrix_bytes, cudaMemcpyHostToDevice, stream);
        apply_nested_polynomial_naive(handle, d_X_naive, rows, cols, polynomials, stream);
    }
    cudaStreamSynchronize(stream);

    float total_time_naive_ms = 0.0f;
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemcpyAsync(d_X_naive, h_X_init.data(), matrix_bytes, cudaMemcpyHostToDevice, stream);

        cudaEventRecord(start, stream);
        apply_nested_polynomial_naive(handle, d_X_naive, rows, cols, polynomials, stream);
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_naive_ms += elapsed_ms;
    }
    float avg_time_naive_ms = total_time_naive_ms / benchmark_runs;
    cudaMemcpy(h_X_naive_result.data(), d_X_naive, matrix_bytes, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------------------
    // 2. Benchmark Optimized Implementation (Fused Diag-Add + Stream Re-use)
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_runs; ++i) {
        cudaMemcpyAsync(d_X_opt, h_X_init.data(), matrix_bytes, cudaMemcpyHostToDevice, stream);
        apply_nested_polynomial(handle, d_X_opt, rows, cols, polynomials, stream);
    }
    cudaStreamSynchronize(stream);

    float total_time_opt_ms = 0.0f;
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemcpyAsync(d_X_opt, h_X_init.data(), matrix_bytes, cudaMemcpyHostToDevice, stream);

        cudaEventRecord(start, stream);
        apply_nested_polynomial(handle, d_X_opt, rows, cols, polynomials, stream);
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_opt_ms += elapsed_ms;
    }
    float avg_time_opt_ms = total_time_opt_ms / benchmark_runs;
    cudaMemcpy(h_X_opt_result.data(), d_X_opt, matrix_bytes, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------------------
    // 3. Mathematical Verification & Metrics
    // -------------------------------------------------------------------------
    matrix_utils::compare::ErrorMetrics<float> metrics = matrix_utils::compare::compute_errors(h_X_naive_result, h_X_opt_result);
    std::cout << "--- Accuracy Verification (Optimized vs Baseline) ---\n";
    print_metrics(metrics);

    // -------------------------------------------------------------------------
    // 4. Performance Verdict
    // -------------------------------------------------------------------------
    std::cout << "\n--- Execution Latency Breakdown ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Naive cuBLAS Routine:     " << avg_time_naive_ms << " ms\n";
    std::cout << "Custom Fused Routine:     " << avg_time_opt_ms   << " ms\n";

    float speedup = avg_time_naive_ms / avg_time_opt_ms;
    std::cout << "Speedup:                  " << speedup << "x ";
    if (speedup > 1.0f) {
        std::cout << "[Custom implementation is FASTER]\n\n";
    } else {
        std::cout << "[Baseline is FASTER]\n\n";
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_X_naive);
    cudaFree(d_X_opt);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return 0;
}
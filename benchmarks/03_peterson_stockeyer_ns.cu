#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/matrix_gen.hpp"
#include "../src/common/paterson_stockmeyer.cuh"
// ---------------------------------------------------------
// Matrix Generation & Utilities
// ---------------------------------------------------------

template<typename T>
void generate_matrix(std::vector<T>& mat, int rows, int cols, T phi = 0.5, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> unif(0.0, 1.0);
    std::normal_distribution<T> norm(0.0, 1.0);

    for(int i = 0; i < rows * cols; ++i) {
        T rand_val = unif(gen);
        T randn_val = norm(gen);
        mat[i] = (rand_val - 0.5) * std::exp(phi * randn_val);
    }
}

// Host-side verification to find maximum absolute error
float calculate_max_error(const std::vector<float>& A, const std::vector<float>& B) {
    float max_err = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        float diff = std::abs(A[i] - B[i]);
        if (diff > max_err) {
            max_err = diff;
        }
    }
    return max_err;
}

// ---------------------------------------------------------
// Benchmark Execution
// ---------------------------------------------------------

int main() {
    // Benchmark configuration
    const int m = 8192;
    const std::vector<int> n_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup_runs = 3;
    const int benchmark_runs = 10;
    const float EPSILON = 1e-3f; // Acceptable FP32 drift for high-degree polynomial

    // Simulating an Order-5 NS iteration (Degree 4 in M) -> 5 coefficients
    const uint degree = 4;
    std::vector<float> h_coeffs = {1.0f, -0.5f, 0.25f, -0.125f, 0.0625f};

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Setup CUDA timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Starting Benchmark (Rows = " << m << ")" << std::endl;
    std::cout << "Rows,Cols,Naive_Time_ms,PS_Time_ms,Max_Error,Match_Status" << std::endl;

    for (int n : n_sizes) {
        size_t matrix_elements = m * n;
        size_t matrix_bytes = matrix_elements * sizeof(float);

        // Host allocations
        std::vector<float> h_X(matrix_elements);
        std::vector<float> h_Res_Naive(matrix_elements);
        std::vector<float> h_Res_PS(matrix_elements);

        // Generate random matrix
        generate_matrix(h_X, m, n, 0.5f, 42);

        // Device allocations
        float *d_X, *d_Res_Naive, *d_Res_PS, *d_coeffs;
        CHECK_CUDA(cudaMalloc(&d_X, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_Res_Naive, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_Res_PS, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_coeffs, h_coeffs.size() * sizeof(float)));

        // Copy initial data to device
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), matrix_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_coeffs, h_coeffs.data(), h_coeffs.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize PS Workspace
        psWorkspace ws(degree, n);

        // ==========================================
        // 1. Benchmark Naive (Factorized Horner)
        // ==========================================
        for (int i = 0; i < warmup_runs; ++i) {
            calculateNS_Naive(handle, d_X, d_Res_Naive, d_coeffs, m, n, degree);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            calculateNS_Naive(handle, d_X, d_Res_Naive, d_coeffs, m, n, degree);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float naive_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));
        naive_ms /= benchmark_runs; // Average per run

        // ==========================================
        // 2. Benchmark Paterson-Stockmeyer
        // ==========================================
        for (int i = 0; i < warmup_runs; ++i) {
            calculateNS_PS(handle, ws, d_X, d_Res_PS, d_coeffs, m);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            calculateNS_PS(handle, ws, d_X, d_Res_PS, d_coeffs, m);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ps_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ps_ms, start, stop));
        ps_ms /= benchmark_runs; // Average per run

        // ==========================================
        // 3. Correctness Verification
        // ==========================================
        CHECK_CUDA(cudaMemcpy(h_Res_Naive.data(), d_Res_Naive, matrix_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_Res_PS.data(), d_Res_PS, matrix_bytes, cudaMemcpyDeviceToHost));

        float max_error = calculate_max_error(h_Res_Naive, h_Res_PS);
        std::string status = (max_error <= EPSILON) ? "PASS" : "FAIL";

        // ==========================================
        // Output CSV Row
        // ==========================================
        std::cout << std::fixed << std::setprecision(4)
                  << m << ","
                  << n << ","
                  << naive_ms << ","
                  << ps_ms << ","
                  << std::scientific << std::setprecision(2) << max_error << ","
                  << status << std::endl;

        // Cleanup for current size
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Res_Naive));
        CHECK_CUDA(cudaFree(d_Res_PS));
        CHECK_CUDA(cudaFree(d_coeffs));
    }

    // Global cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
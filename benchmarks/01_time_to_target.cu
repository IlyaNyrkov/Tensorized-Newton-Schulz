#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cans/cans.cuh"
#include "../utils/standard_newton_schulz.cuh"
#include "../utils/matrix_gen.hpp"

// -----------------------------------------------------------------------------
// Helper: Compute Max Absolute Error of |X^T X - I|
// -----------------------------------------------------------------------------
float evaluate_orthogonality_error(cublasHandle_t handle, float* d_X, float* d_M, int n, int p, cudaStream_t stream) {
    float alpha_1 = 1.0f, beta_0 = 0.0f;

    // M = X^T X (Column-major mapping to safely handle tall matrices)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n,
                &alpha_1, d_X, p, d_X, p, &beta_0, d_M, p);

    std::vector<float> h_M(p * p);
    cudaMemcpyAsync(h_M.data(), d_M, p * p * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float max_err = 0.0f;
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float actual = h_M[i * p + j];
            float err = std::abs(actual - expected);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// -----------------------------------------------------------------------------
// Main Benchmark Runner
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    float condition_number = (argc > 1) ? std::stof(argv[1]) : 5.0f;

    const int WARMUPS = 3;
    const int RUNS = 10;
    const int STD_NS_MAX_ITER = 200;

    std::vector<std::pair<size_t, size_t>> matrix_sizes = {
            {8192, 64},
            {8192, 128},
            {8192, 256},
            {8192, 512},
            {8192, 1024},
            {8192, 2048},
            {8192, 4096},
            {8192, 8192},
            {16384, 64},
            {16384, 128},
            {16384, 256},
            {16384, 512},
            {16384, 1024},
            {16384, 2048},
            {16384, 4096},
            {16384, 8192},
            {16384, 16384},
    };

    std::vector<float> target_deltas = {0.00188f, 0.00350f, 0.00443f, 0.01000f, 0.05000f, 0.10000f, 0.20000f, 0.30000f};

    std::cout << "Initializing Benchmark: Time-to-Target (Aspect Ratio Scaling)\n";
    std::cout << "Matrix Condition number: " << condition_number << "\n";
    std::cout << "Warmups: " << WARMUPS << " | Timed Runs: " << RUNS << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Rows, Cols, Target_Delta, CANS_Error, CANS_Time_ms, CANS_TFLOPS, Std_NS_Iters, Std_NS_Time_ms, Std_NS_TFLOPS, Speedup\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t temp_handle;
    cublasCreate(&temp_handle);
    cublasSetStream(temp_handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    struct Result {
        size_t rows;
        size_t cols;
        float delta;
        float error;
        float cans_time;
        double cans_tflops;
        int std_iters;
        float std_time;
        double std_tflops;
    };
    std::vector<Result> results;

    for (const auto& size : matrix_sizes) {
        size_t n = size.first;
        size_t p = size.second;

        std::vector<float> h_X_orig = matrix_utils::generation::generate_conditioned_matrix<float>(n, p, condition_number, 42);

        float *d_X, *d_X_std, *d_X_std_next, *d_M_std, *d_scalar_std;
        cudaMalloc(&d_X, n * p * sizeof(float));
        cudaMalloc(&d_X_std, n * p * sizeof(float));
        cudaMalloc(&d_X_std_next, n * p * sizeof(float));
        cudaMalloc(&d_M_std, p * p * sizeof(float));
        cudaMalloc(&d_scalar_std, 2 * sizeof(float));

        for (float delta : target_deltas) {
            // =====================================================================
            // 1. CANS Execution & Error Evaluation
            // =====================================================================
            cans::Workspace<float> ws(n, p, delta, stream);

            cudaMemcpyAsync(d_X, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
            cans::cans_orthogonalize(ws, d_X);
            float cans_actual_err = evaluate_orthogonality_error(ws.handle, d_X, ws.d_M, n, p, stream);

            for(int i = 0; i < WARMUPS; ++i) {
                cudaMemcpyAsync(d_X, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                cans::cans_orthogonalize(ws, d_X);
            }
            cudaStreamSynchronize(stream);

            float cans_total_ms = 0.0f;
            for(int i = 0; i < RUNS; ++i) {
                cudaMemcpyAsync(d_X, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                cudaEventRecord(start, stream);
                cans::cans_orthogonalize(ws, d_X);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms; cudaEventElapsedTime(&ms, start, stop);
                cans_total_ms += ms;
            }
            float cans_avg_ms = cans_total_ms / RUNS;

            // Calculate CANS TFLOPS
            // Phase 0: 2*N*P^2 | Phase 1: ~10*(2*P^3) | Phase 2: Loops*(4*N*P^2)
            double cans_flops = (2.0 * n * p * p) +
                                (10.0 * 2.0 * p * p * p) +
                                (ws.phase2_iterations.size() * 4.0 * n * p * p);
            double cans_tflops = cans_flops / (cans_avg_ms * 1e6); // 1e6 converts ms to seconds * 1e12 (TFLOPS)

            // =====================================================================
            // 2. Standard NS: Find Iteration Threshold
            // =====================================================================
            cudaMemcpyAsync(d_X_std, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
            cans_utils::reference::apply_standard_normalization(temp_handle, d_X_std, d_scalar_std, n, p, stream);

            int required_std_iters = 0;
            float std_err = 1.0f;

            for (int k = 1; k <= STD_NS_MAX_ITER; ++k) {
                cans_utils::reference::apply_standard_ns_iterations(temp_handle, d_X_std, d_X_std_next, d_M_std, n, p, 1, stream);
                std_err = evaluate_orthogonality_error(temp_handle, d_X_std, d_M_std, n, p, stream);
                if (std_err <= cans_actual_err) {
                    required_std_iters = k;
                    break;
                }
            }

            if (required_std_iters == 0) required_std_iters = STD_NS_MAX_ITER;

            // =====================================================================
            // 3. Benchmark Standard NS
            // =====================================================================
            auto run_std_ns = [&]() {
                cans_utils::reference::apply_standard_normalization(temp_handle, d_X_std, d_scalar_std, n, p, stream);
                cans_utils::reference::apply_standard_ns_iterations(temp_handle, d_X_std, d_X_std_next, d_M_std, n, p, required_std_iters, stream);
            };

            for(int i = 0; i < WARMUPS; ++i) {
                cudaMemcpyAsync(d_X_std, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                run_std_ns();
            }
            cudaStreamSynchronize(stream);

            float std_total_ms = 0.0f;
            for(int i = 0; i < RUNS; ++i) {
                cudaMemcpyAsync(d_X_std, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                cudaEventRecord(start, stream);
                run_std_ns();
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms; cudaEventElapsedTime(&ms, start, stop);
                std_total_ms += ms;
            }
            float std_avg_ms = std_total_ms / RUNS;

            // Calculate Standard NS TFLOPS
            // Phase 0: 2*N*P^2 | Iterations: Iters*(4*N*P^2)
            double std_flops = (2.0 * n * p * p) +
                               (required_std_iters * 4.0 * n * p * p);
            double std_tflops = std_flops / (std_avg_ms * 1e6);

            // Print CSV Row
            float speedup = std_avg_ms / cans_avg_ms;
            std::cout << std::fixed << std::setprecision(5)
                      << n << ", " << p << ", "
                      << delta << ", "
                      << std::scientific << cans_actual_err << std::fixed << ", "
                      << cans_avg_ms << ", "
                      << cans_tflops << ", "
                      << required_std_iters << ", "
                      << std_avg_ms << ", "
                      << std_tflops << ", "
                      << speedup << "x\n";

            results.push_back({n, p, delta, cans_actual_err, cans_avg_ms, cans_tflops, required_std_iters, std_avg_ms, std_tflops});
        }

        cudaFree(d_X); cudaFree(d_X_std); cudaFree(d_X_std_next); cudaFree(d_M_std); cudaFree(d_scalar_std);
    }

    // =====================================================================
    // 4. Summary Output
    // =====================================================================
    std::cout << "\n========================================================\n";
    std::cout << " SUMMARY: CANS vs Standard Newton-Schulz by Matrix Size\n";
    std::cout << "========================================================\n";

    size_t current_rows = 0;
    for (const auto& r : results) {
        if (r.rows != current_rows) {
            std::cout << "\n--- Matrix Shape: " << r.rows << " x " << r.cols << " ---\n";
            current_rows = r.rows;
        }
        float speedup = r.std_time / r.cans_time;
        std::cout << "Target Delta " << std::fixed << std::setprecision(5) << r.delta << ":\n";
        std::cout << "  - CANS achieved error " << std::scientific << r.error << std::fixed
                  << " in " << r.cans_time << " ms (" << std::setprecision(2) << r.cans_tflops << " TFLOPS).\n";
        std::cout << "  - Std NS required " << r.std_iters << " iterations to match this error, taking "
                  << r.std_time << " ms (" << std::setprecision(2) << r.std_tflops << " TFLOPS).\n";
        std::cout << "  - Verdict: CANS is " << speedup << "x faster.\n";
    }

    cublasDestroy(temp_handle); cudaStreamDestroy(stream);
    return 0;
}
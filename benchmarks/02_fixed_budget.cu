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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n, &alpha_1, d_X, p, d_X, p, &beta_0, d_M, p);

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
int main() {
    const size_t n = 4096;
    const size_t p = 4096;
    const int TRIALS = 10;

    std::vector<float> condition_numbers = {1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 100.0f, 200.0f, 300.0f, 500.0f, 700.0f, 1000.0f, 2000.0f, 3000.0f, 5000.0f, 7000.0f, 10000.0f};
    std::vector<float> target_deltas = {0.00188f, 0.00350f, 0.00443f, 0.01000f, 0.05000f, 0.10000f, 0.20000f, 0.30000f};

    std::cout << "Initializing Benchmark: Fixed-Compute Budget (Multi-Trial)\n";
    std::cout << "Matrix Dimensions: " << n << " x " << p << "\n";
    std::cout << "Trials per Configuration: " << TRIALS << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Kappa, Trial_ID, Target_Delta, Total_GEMMs, CANS_Error, Std_NS_Error\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t temp_handle;
    cublasCreate(&temp_handle);
    cublasSetStream(temp_handle, stream);

    // Allocate device buffers ONCE to prevent OOM
    float *d_X, *d_X_std, *d_X_std_next, *d_M_std, *d_scalar_std;
    cudaMalloc(&d_X, n * p * sizeof(float));
    cudaMalloc(&d_X_std, n * p * sizeof(float));
    cudaMalloc(&d_X_std_next, n * p * sizeof(float));
    cudaMalloc(&d_M_std, p * p * sizeof(float));
    cudaMalloc(&d_scalar_std, 2 * sizeof(float));

    for (float kappa : condition_numbers) {
        for (int trial = 0; trial < TRIALS; ++trial) {

            // Generate a uniquely seeded matrix for this specific trial
            int seed = 42 + trial;
            std::vector<float> h_X_orig = matrix_utils::generation::generate_conditioned_matrix<float>(n, p, kappa, seed);

            for (float delta : target_deltas) {

                // 1. Initialize CANS to extract the mathematical execution graph
                cans::Workspace<float> ws(n, p, delta, stream);

                int phase1_gemms = 0;
                for (const auto& poly : ws.phase1_polynomials) {
                    phase1_gemms += poly.size();
                }
                int total_cans_gemms = 1 + phase1_gemms + (ws.phase2_iterations.size() * 2);

                // 2. Execute CANS
                cudaMemcpyAsync(d_X, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                cans::cans_orthogonalize(ws, d_X);
                float cans_err = evaluate_orthogonality_error(ws.handle, d_X, ws.d_M, n, p, stream);

                // 3. Match Standard NS Compute Budget
                int std_ns_iters = (total_cans_gemms - 1) / 2;

                cudaMemcpyAsync(d_X_std, h_X_orig.data(), n * p * sizeof(float), cudaMemcpyHostToDevice, stream);
                cans_utils::reference::apply_standard_normalization(temp_handle, d_X_std, d_scalar_std, n, p, stream);
                cans_utils::reference::apply_standard_ns_iterations(temp_handle, d_X_std, d_X_std_next, d_M_std, n, p, std_ns_iters, stream);
                float std_err = evaluate_orthogonality_error(temp_handle, d_X_std, d_M_std, n, p, stream);

                // Print CSV Row
                std::cout << std::fixed << std::setprecision(2) << kappa << ", "
                          << trial << ", "
                          << std::setprecision(5) << delta << ", "
                          << total_cans_gemms << ", "
                          << std::scientific << cans_err << ", "
                          << std_err << "\n";
            }
        }
    }

    cudaFree(d_X); cudaFree(d_X_std); cudaFree(d_X_std_next); cudaFree(d_M_std); cudaFree(d_scalar_std);
    cublasDestroy(temp_handle); cudaStreamDestroy(stream);

    return 0;
}
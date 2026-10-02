#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../src/common/paterson_stockmeyer.cuh"

template<typename T>
void generate_normalized_matrix(std::vector<T>& mat, int rows, int cols, T phi = 0.5, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> unif(0.0, 1.0);
    std::normal_distribution<T> norm(0.0, 1.0);

    double frob_sq = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        T val = (unif(gen) - 0.5f) * std::exp(phi * norm(gen));
        mat[i] = val;
        frob_sq += (double)val * val;
    }

    // Scale X so ||X||_2 <= ||X||_F = 1.0 (Strictly bounds eigenvalues of M in [0, 1])
    float inv_norm = 1.0f / (float)std::sqrt(frob_sq);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] *= inv_norm;
    }
}

float calculate_max_relative_error(const std::vector<float>& A, const std::vector<float>& B) {
    float max_err = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        float diff = std::abs(A[i] - B[i]);
        float denom = std::max(std::abs(A[i]), 1e-6f);
        max_err = std::max(max_err, diff / denom);
    }
    return max_err;
}

int main() {
    const int m = 8192;
    const std::vector<int> n_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup_runs = 3;
    const int benchmark_runs = 10;
    const float EPSILON = 5e-3f; // 0.5% max relative tolerance for 5-term polynomial

    // Standard normalized NS coefficients (degree 4 in M -> Order-5 NS)
    const uint degree = 4;
    std::vector<float> h_coeffs = {1.0f, -0.5f, 0.375f, -0.3125f, 0.2734f};

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Starting Benchmark (Rows = " << m << ")" << std::endl;
    std::cout << "Degree " << degree + 1 << "Newton Schulz" << std::endl;
    std::cout << "Rows,Cols,Naive_Time_ms,PS_Time_ms,Max_Rel_Error,Match_Status" << std::endl;

    for (int n : n_sizes) {
        size_t matrix_elements = (size_t)m * n;
        size_t matrix_bytes = matrix_elements * sizeof(float);

        std::vector<float> h_X(matrix_elements);
        std::vector<float> h_Res_Naive(matrix_elements);
        std::vector<float> h_Res_PS(matrix_elements);

        generate_normalized_matrix(h_X, m, n, 0.5f, 42);

        float *d_X, *d_Res_Naive, *d_Res_PS, *d_coeffs;
        CHECK_CUDA(cudaMalloc(&d_X, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_Res_Naive, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_Res_PS, matrix_bytes));
        CHECK_CUDA(cudaMalloc(&d_coeffs, h_coeffs.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), matrix_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_coeffs, h_coeffs.data(), h_coeffs.size() * sizeof(float), cudaMemcpyHostToDevice));

        psWorkspace ws(degree, n);
        naiveWorkspace n_ws(n);

        // 1. Benchmark Naive
        for (int i = 0; i < warmup_runs; ++i) {
            calculateNS_Naive(handle, n_ws, d_X, d_Res_Naive, h_coeffs, m, n, degree);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            calculateNS_Naive(handle, n_ws, d_X, d_Res_Naive, h_coeffs, m, n, degree);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float naive_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));
        naive_ms /= benchmark_runs;

        // 2. Benchmark Paterson-Stockmeyer
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
        ps_ms /= benchmark_runs;

        // 3. Verification
        CHECK_CUDA(cudaMemcpy(h_Res_Naive.data(), d_Res_Naive, matrix_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_Res_PS.data(), d_Res_PS, matrix_bytes, cudaMemcpyDeviceToHost));

        float rel_error = calculate_max_relative_error(h_Res_Naive, h_Res_PS);
        std::string status = (rel_error <= EPSILON) ? "PASS" : "FAIL";

        std::cout << std::fixed << std::setprecision(4)
                  << m << ","
                  << n << ","
                  << naive_ms << ","
                  << ps_ms << ","
                  << std::scientific << std::setprecision(2) << rel_error << ","
                  << status << std::endl;

        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Res_Naive));
        CHECK_CUDA(cudaFree(d_Res_PS));
        CHECK_CUDA(cudaFree(d_coeffs));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
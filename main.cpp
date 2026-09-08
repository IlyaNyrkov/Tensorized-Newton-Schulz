#include <vector>
#include <cuda.h>
#include <stdexcept>
#include <cublas_v2.h>

__global__ void scale_add_diag_kernel(const float* M, float* out, float alpha, float beta, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p * p) {
        int row = idx % p;
        int col = idx / p;
        // out = alpha * M + beta * I
        float val = alpha * M[idx];
        if (row == col) {
            val += beta;
        }
        out[idx] = val;
    }
}

// Host launcher
void launch_scale_add_diag(const float* d_in, float* d_out, float alpha, float beta, int p, cudaStream_t stream) {
    int threads = 256;
    int blocks = (p * p + threads - 1) / threads;
    scale_add_diag_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, alpha, beta, p);
}

void apply_degree_3_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, const float* coeffs, int n, int p, cudaStream_t stream) {
    float alpha_1 = 1.0f, beta_0 = 0.0f;

    // 1. M = X^T * X
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X, n, &beta_0, d_M, p);

    // 2. Q = c_3 * M + c_1 * I (Stored in d_temp1)
    launch_scale_add_diag(d_M, d_temp1, coeffs[1], coeffs[0], p, stream);

    // 3. X_out = X * Q
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n, d_temp1, p, &beta_0, d_X_out, n);

    // 4. In-place update (In a real training loop, swap pointers instead of copying)
    cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void apply_degree_5_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, float* d_temp2, const float* coeffs, int n, int p, cudaStream_t stream) {
    float alpha_1 = 1.0f, beta_0 = 0.0f;

    // 1. M = X^T * X
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X, n, &beta_0, d_M, p);

    // 2. K = c_5 * M + c_3 * I (Stored in d_temp1)
    launch_scale_add_diag(d_M, d_temp1, coeffs[2], coeffs[1], p, stream);

    // 3. Q = M * K (Stored in d_temp2 to avoid aliasing)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p, d_temp1, p, &beta_0, d_temp2, p);

    // 4. Q = 1.0 * Q + c_1 * I (In-place kernel update on d_temp2)
    launch_scale_add_diag(d_temp2, d_temp2, 1.0f, coeffs[0], p, stream);

    // 5. X_out = X * Q
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n, d_temp2, p, &beta_0, d_X_out, n);

    cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void apply_degree_7_poly(cublasHandle_t handle, float* d_X, float* d_X_out, float* d_M, float* d_temp1, float* d_temp2, const float* coeffs, int n, int p, cudaStream_t stream) {
    float alpha_1 = 1.0f, beta_0 = 0.0f;

    // 1. M = X^T * X
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha_1, d_X, n, d_X, n, &beta_0, d_M, p);

    // 2. L = c_7 * M + c_5 * I (Stored in d_temp1)
    launch_scale_add_diag(d_M, d_temp1, coeffs[3], coeffs[2], p, stream);

    // 3. K = M * L (Stored in d_temp2)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p, d_temp1, p, &beta_0, d_temp2, p);

    // 4. K = 1.0 * K + c_3 * I (In-place on d_temp2)
    launch_scale_add_diag(d_temp2, d_temp2, 1.0f, coeffs[1], p, stream);

    // 5. Q = M * K (Stored back in d_temp1)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, p, &alpha_1, d_M, p, d_temp2, p, &beta_0, d_temp1, p);

    // 6. Q = 1.0 * Q + c_1 * I (In-place on d_temp1)
    launch_scale_add_diag(d_temp1, d_temp1, 1.0f, coeffs[0], p, stream);

    // 7. X_out = X * Q
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, p, &alpha_1, d_X, n, d_temp1, p, &beta_0, d_X_out, n);

    cudaMemcpyAsync(d_X, d_X_out, n * p * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

template <typename T>
void apply_nested_polynomial(cublasHandle_t handle, T* d_X, int n, int p, const std::vector<std::vector<float>>& polynomials, cudaStream_t stream = nullptr) {

    // 1. Allocate maximum required workspace upfront to prevent loop-level allocations
    size_t x_out_elems = n * p;
    size_t square_elems = p * p;

    T* d_workspace;
    size_t total_bytes = (x_out_elems + 3 * square_elems) * sizeof(T);
    cudaMallocAsync(&d_workspace, total_bytes, stream);

    // Map pointers into the contiguous block
    T* d_X_out = d_workspace;
    T* d_M     = d_workspace + x_out_elems;
    T* d_temp1 = d_M + square_elems;
    T* d_temp2 = d_temp1 + square_elems;

    // 2. Generalized loop driving specialized kernels
    for (const auto& coeffs : polynomials) {

        switch (coeffs.size()) {
            case 2:
                apply_degree_3_poly(handle, d_X, d_X_out, d_M, d_temp1, coeffs.data(), n, p, stream);
                break;
            case 3:
                apply_degree_5_poly(handle, d_X, d_X_out, d_M, d_temp1, d_temp2, coeffs.data(), n, p, stream);
                break;
            case 4:
                apply_degree_7_poly(handle, d_X, d_X_out, d_M, d_temp1, d_temp2, coeffs.data(), n, p, stream);
                break;
            default:
                cudaFreeAsync(d_workspace, stream);
                throw std::runtime_error("Unsupported polynomial coefficient count.");
        }
    }

    // 3. Asynchronous free
    cudaFreeAsync(d_workspace, stream);
}

int main() {

}
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

namespace cans {
// 1-thread kernel to compute the final scaling factor entirely on the device
    __global__ void compute_gelfand_scale_kernel(const float *d_frobenius_norm, float *d_scale_factor) {
        // Diagram formula: C = sqrt(||X^T X||_F)
        // We need to scale X by 1/C
        float norm = *d_frobenius_norm;

        // Guard against division by zero for edge cases
        if (norm > 1e-12f) {
            *d_scale_factor = 1.0f / sqrtf(norm);
        } else {
            *d_scale_factor = 1.0f;
        }
    }

    void apply_gelfand_normalization(cublasHandle_t handle, float *d_X, float *d_M, float *d_scalar_workspace, int n, int p,
                                cudaStream_t stream) {
        float alpha_1 = 1.0f, beta_0 = 0.0f;

        // 1. Compute M = X^T * X (Compute bound)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n,
                    &alpha_1, d_X, p, d_X, p, &beta_0, d_M, p);

        // 2. Set cuBLAS to write scalar results directly to device memory
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

        // 3. Compute Frobenius norm (L2 norm of flattened M array)
        // d_scalar_workspace[0] will hold ||X^T X||_F
        cublasSnrm2(handle, p * p, d_M, 1, d_scalar_workspace);

        // 4. Compute 1 / sqrt(C) using a single GPU thread
        // d_scalar_workspace[1] will hold the final scale factor
        float *d_norm = d_scalar_workspace;
        float *d_scale = d_scalar_workspace + 1;
        compute_gelfand_scale_kernel<<<1, 1, 0, stream>>>(d_norm, d_scale);

        // 5. In-place element-wise scaling of X using the device-side scalar
        cublasSscal(handle, n * p, d_scale, d_X, 1);

        // 6. Restore host pointer mode for subsequent API calls
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    }
}
#pragma once

#include "workspace.hpp"
#include "../../src/common/gelfand_normalization.cuh"
#include "../../src/common/nested_polynomial.cuh"
#include "../../src/common/cans_newton_iter.cuh"

namespace cans {

     void cans_orthogonalize(Workspace<float>& ws, float* d_X, CansTimings* timings = nullptr) {

        if (timings) cudaEventRecord(timings->events[0], ws.stream);

        apply_gelfand_normalization(ws.handle, d_X, ws.d_M, ws.d_scalar_ws, ws.n, ws.p, ws.stream);

        if (timings) cudaEventRecord(timings->events[1], ws.stream);

        apply_nested_polynomial(ws.handle, d_X, ws.n, ws.p, ws.phase1_polynomials, ws.stream);

        if (timings) cudaEventRecord(timings->events[2], ws.stream);

        apply_cans_pure_blas(ws.handle, d_X, ws.n, ws.p, ws.phase2_iterations, ws.stream);

        if (timings) cudaEventRecord(timings->events[3], ws.stream);

        if (timings) {
            cudaEventSynchronize(timings->events[3]);
            cudaEventElapsedTime(&timings->phase0_ms, timings->events[0], timings->events[1]);
            cudaEventElapsedTime(&timings->phase1_ms, timings->events[1], timings->events[2]);
            cudaEventElapsedTime(&timings->phase2_ms, timings->events[2], timings->events[3]);
            cudaEventElapsedTime(&timings->total_ms,  timings->events[0], timings->events[3]);
        }
    }

} // namespace cans
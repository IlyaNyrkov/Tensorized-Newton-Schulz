#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cans::lookup_table {

// Define the tolerance for matching requested floating-point Deltas
    constexpr float DELTA_TOLERANCE = 1e-4f;

// -------------------------------------------------------------------------
// 1. Constexpr Storage Structures
// -------------------------------------------------------------------------
    struct RawCansConfig {
        float delta;
        int num_phase1_steps;
        std::array<std::array<float, 4>, 3>
                phase1_polys;  // Max 3 polys, Max 4 coeffs (Deg 7)
        int num_phase2_steps;
        std::array<std::array<float, 2>, 4>
                phase2_iters;  // Max 4 loops, 2 coeffs (c1, c3)
    };

    // Available deltas
    // [0.00188,
    constexpr std::array<RawCansConfig, 8> CONFIG_REGISTRY = {
            {// Delta = 0.00188
                    {0.00188f,
                     3,
                     {{{6.85849136f, -18.76629940f, 13.59237477f, 0.0f},
                       {2.59339089f, -1.92110886f, 0.44640659f, 0.0f},
                       {2.26070146f, -2.34050287f, 1.39706530f, 0.0f}}},
                     2,
                     {{{1.50000309f, 0.50000044f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.00350
                    {0.00350f,
                     3,
                     {{{4.52756225f, -4.31303553f, 0.0f, 0.0f},
                       {3.57085941f, -4.85909450f, 2.54383086f, -0.41762005f},
                       {2.28876028f, -2.39901924f, 1.42849339f, -0.32153688f}}},
                     2,
                     {{{1.50001072f, 0.50000153f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.00443
                    {0.00443f,
                     3,
                     {{{4.55741422f, -4.35240249f, 0.0f, 0.0f},
                       {3.63058296f, -4.96518992f, 2.58740030f, -0.42160228f},
                       {2.30216330f, -2.42692774f, 1.44332413f, -0.32270912f}}},
                     2,
                     {{{1.50001717f, 0.50000245f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.01
                    {0.01000f,
                     3,
                     {{{6.89922738f, -18.92583406f, 13.71882326f, 0.0f},
                       {2.61575740f, -1.93965514f, 0.44845902f, 0.0f},
                       {1.94444905f, -1.32517537f, 0.38259337f, 0.0f}}},
                     2,
                     {{{1.50008750f, 0.50001250f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.05
                    {0.05000f,
                     3,
                     {{{4.58046835f, -4.38280929f, 0.0f, 0.0f},
                       {3.01166051f, -2.25571391f, 0.48376556f, 0.0f},
                       {2.09160813f, -1.47403557f, 0.39801761f, 0.0f}}},
                     3,
                     {{{1.50218867f, 0.50031263f},
                       {1.50000308f, 0.50000044f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.1
                    {0.10000f,
                     2,
                     {{{9.10367418f, -48.57678443f, 87.21510564f, -47.36880449f},
                       {2.87067106f, -3.56634485f, 1.99146557f, -0.36832480f},
                       {0.0f, 0.0f, 0.0f, 0.0f}}},
                     3,
                     {{{1.50876878f, 0.50125209f},
                       {1.50004949f, 0.50000707f},
                       {1.50000000f, 0.50000000f},
                       {0.0f, 0.0f}}}},

                    // Delta = 0.2
                    {0.20000f,
                     2,
                     {{{7.10481318f, -19.73166398f, 14.35777956f, 0.0f},
                       {3.27415277f, -4.32326898f, 2.32029435f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f}}},
                     4,
                     {{{1.53530196f, 0.50503354f},
                       {1.50080538f, 0.50011505f},
                       {1.50000042f, 0.50000006f},
                       {1.50000000f, 0.50000000f}}}},

                    // Delta = 0.3
                    {0.30000f,
                     2,
                     {{{4.56167242f, -4.35801849f, 0.0f, 0.0f},
                       {3.63932233f, -4.98066801f, 2.59373938f, -0.42218254f},
                       {0.0f, 0.0f, 0.0f, 0.0f}}},
                     4,
                     {{{1.58029120f, 0.51142110f},
                       {1.50419601f, 0.50059929f},
                       {1.50001133f, 0.50000162f},
                       {1.50000000f, 0.50000000f}}}}}};

// -------------------------------------------------------------------------
// 2. Runtime API Structures
// -------------------------------------------------------------------------
    struct CansConfig {
        float delta;
        std::vector<std::vector<float>> phase1_polynomials;
        std::vector<std::array<float, 2>> phase2_iterations;
    };

// -------------------------------------------------------------------------
// 3. Lookup & Parsing Utility
// -------------------------------------------------------------------------
    inline CansConfig get_config(float target_delta) {
        for (const auto& raw_cfg : CONFIG_REGISTRY) {
            // Check if delta is within the defined tolerance
            if (std::abs(raw_cfg.delta - target_delta) <= DELTA_TOLERANCE) {
                CansConfig parsed_cfg;
                parsed_cfg.delta = raw_cfg.delta;

                // Extract Phase 1 Polynomials and strip padded zeros
                for (int i = 0; i < raw_cfg.num_phase1_steps; ++i) {
                    std::vector<float> poly;
                    for (int j = 0; j < 4; ++j) {
                        float val = raw_cfg.phase1_polys[i][j];
                        if (val != 0.0f) {
                            poly.push_back(val);
                        }
                    }
                    parsed_cfg.phase1_polynomials.push_back(poly);
                }

                // Extract Phase 2 Iterations
                for (int i = 0; i < raw_cfg.num_phase2_steps; ++i) {
                    parsed_cfg.phase2_iterations.push_back(raw_cfg.phase2_iters[i]);
                }

                return parsed_cfg;
            }
        }

        throw std::invalid_argument("CANS Configuration not found for Delta: " +
                                    std::to_string(target_delta));
    }

}  // namespace cans_constants
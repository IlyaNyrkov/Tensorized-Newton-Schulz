#pragma once
#include <array>

namespace cans_constants {

    // Configuration for Delta = 0.00188
    // Phase 1 Sequence: [5, 5, 7], Cost: 10 GEMMs, Achieved 'a': 0.046262
    constexpr std::array<std::array<float, 3>, 3> P_0_00188 = {{
        {6.85849136f, -18.76629940f, 13.59237477f},
        {2.59339089f, -1.92110886f, 0.44640659f},
        {2.26070146f, -2.34050287f, 1.39706530f}
    }};
    // Phase 2 Iterations: 2 loops
    constexpr std::array<std::array<float, 2>, 2> C_0_00188 = {{
        {1.50000309f, 0.50000044f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.0035
    // Phase 1 Sequence: [3, 7, 7], Cost: 10 GEMMs, Achieved 'a': 0.047485
    constexpr std::array<std::array<float, 4>, 3> P_0_0035 = {{
        {4.52756225f, -4.31303553f, 0.0f, 0.0f},
        {3.57085941f, -4.85909450f, 2.54383086f, -0.41762005f},
        {2.28876028f, -2.39901924f, 1.42849339f, -0.32153688f}
    }};
    // Phase 2 Iterations: 2 loops
    constexpr std::array<std::array<float, 2>, 2> C_0_0035 = {{
        {1.50001072f, 0.50000153f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.00443
    // Phase 1 Sequence: [3, 7, 7], Cost: 10 GEMMs, Achieved 'a': 0.045072
    constexpr std::array<std::array<float, 4>, 3> P_0_00443 = {{
        {4.55741422f, -4.35240249f, 0.0f, 0.0f},
        {3.63058296f, -4.96518992f, 2.58740030f, -0.42160228f},
        {2.30216330f, -2.42692774f, 1.44332413f, -0.32270912f}
    }};
    // Phase 2 Iterations: 2 loops
    constexpr std::array<std::array<float, 2>, 2> C_0_00443 = {{
        {1.50001717f, 0.50000245f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.05
    // Phase 1 Sequence: [3, 5, 5], Cost: 8 GEMMs, Achieved 'a': 0.043230
    constexpr std::array<std::array<float, 3>, 3> P_0_05 = {{
        {4.58046835f, -4.38280929f, 0.0f},
        {3.01166051f, -2.25571391f, 0.48376556f},
        {2.09160813f, -1.47403557f, 0.39801761f}
    }};
    // Phase 2 Iterations: 3 loops
    constexpr std::array<std::array<float, 2>, 3> C_0_05 = {{
        {1.50218867f, 0.50031263f},
        {1.50000308f, 0.50000044f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.01
    // Phase 1 Sequence: [5, 5, 5], Cost: 9 GEMMs, Achieved 'a': 0.044859
    constexpr std::array<std::array<float, 3>, 3> P_0_01 = {{
        {6.89922738f, -18.92583406f, 13.71882326f},
        {2.61575740f, -1.93965514f, 0.44845902f},
        {1.94444905f, -1.32517537f, 0.38259337f}
    }};
    // Phase 2 Iterations: 2 loops
    constexpr std::array<std::array<float, 2>, 2> C_0_01 = {{
        {1.50008750f, 0.50001250f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.1
    // Phase 1 Sequence: [7, 7], Cost: 8 GEMMs, Achieved 'a': 0.041370
    constexpr std::array<std::array<float, 4>, 2> P_0_1 = {{
        {9.10367418f, -48.57678443f, 87.21510564f, -47.36880449f},
        {2.87067106f, -3.56634485f, 1.99146557f, -0.36832480f}
    }};
    // Phase 2 Iterations: 3 loops
    constexpr std::array<std::array<float, 2>, 3> C_0_1 = {{
        {1.50876878f, 0.50125209f},
        {1.50004949f, 0.50000707f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.2
    // Phase 1 Sequence: [5, 7], Cost: 7 GEMMs, Achieved 'a': 0.038024
    constexpr std::array<std::array<float, 3>, 2> P_0_2 = {{
        {7.10481318f, -19.73166398f, 14.35777956f},
        {3.27415277f, -4.32326898f, 2.32029435f}
    }};
    // Phase 2 Iterations: 4 loops
    constexpr std::array<std::array<float, 2>, 4> C_0_2 = {{
        {1.53530196f, 0.50503354f},
        {1.50080538f, 0.50011505f},
        {1.50000042f, 0.50000006f},
        {1.50000000f, 0.50000000f}
    }};

    // Configuration for Delta = 0.3
    // Phase 1 Sequence: [3, 7], Cost: 6 GEMMs, Achieved 'a': 0.044730
    constexpr std::array<std::array<float, 4>, 2> P_0_3 = {{
        {4.56167242f, -4.35801849f, 0.0f, 0.0f},
        {3.63932233f, -4.98066801f, 2.59373938f, -0.42218254f}
    }};
    // Phase 2 Iterations: 4 loops
    constexpr std::array<std::array<float, 2>, 4> C_0_3 = {{
        {1.58029120f, 0.51142110f},
        {1.50419601f, 0.50059929f},
        {1.50001133f, 0.50000162f},
        {1.50000000f, 0.50000000f}
    }};
}

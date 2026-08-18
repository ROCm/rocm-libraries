// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

// Unit tests for the ConvDepthwiseDirect solver (RDNA wave32 VALU depthwise).
//
// Coverage spans both memory layouts and both dimensionalities, since the solver
// serves each with a distinct device core:
//   - channel-last  (NHWC / NDHWC): the native VALU layout — floor + halo/LDS
//     variants (v2_wstrip_core_nhwc/ndhwc, v3a_microtile, v4_fused, v3b_lds).
//   - channel-first (NCHW / NCDHW): the WStrip floor only, via the layout-mirror
//     cores (v2_core_nchw / v2_core_ncdhw) that coalesce on the contiguous width
//     axis. The halo/LDS variants are channel-last native and decline here, so
//     these cases exercise the floor path exclusively.
//
// The 2D cases pick kernel sizes that route to each variant (3x3 microtile,
// 5x5/7x7 fused, 9x9 LDS) plus stride!=1 / 11x11 cases that fall through to the
// floor; the channel-first cases all route to the floor by design. Verification
// is against the harness's fp64-accumulate naive reference (default ref path).

namespace {

using TestCase = miopen::unit_tests::ConvTestCase;

// A depthwise conv: group_count == C, weights [C, 1, (kd,) kh, kw]. The lens are
// given in logical NCHW / NCDHW order; the layout tag selects the storage order.

// -------- 2D: channel-last (NHWC) — exercises every RDNA variant ------------
auto GetConvTestCasesNHWC2D(miopenDataType_t dt)
{
    return std::vector{
        // clang-format off
        // 3x3 s1 -> microtile
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 3, 3}},
                 dt, {{1, 1}, {1, 1}, {1, 1}, 8}},
        // 5x5 s1 -> fused (plane >= 8 -> 16x16x32 r4x4)
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 5, 5}},
                 dt, {{2, 2}, {1, 1}, {1, 1}, 8}},
        // 7x7 s1 -> fused
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 7, 7}},
                 dt, {{3, 3}, {1, 1}, {1, 1}, 8}},
        // 9x9 s1 -> LDS (register-light; v4's patch would spill)
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 9, 9}},
                 dt, {{4, 4}, {1, 1}, {1, 1}, 8}},
        // 3x3 s2 -> floor (halo variants require s==1)
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 3, 3}},
                 dt, {{1, 1}, {2, 2}, {1, 1}, 8}},
        // 11x11 s1 -> floor (no specialized variant for this size)
        TestCase{{dt, miopenTensorNHWC, {2, 8, 16, 16}}, {dt, miopenTensorNHWC, {8, 1, 11, 11}},
                 dt, {{5, 5}, {1, 1}, {1, 1}, 8}},
        // clang-format on
    };
}

// -------- 2D: channel-first (NCHW) — the new capability, floor path ---------
auto GetConvTestCasesNCHW2D(miopenDataType_t dt)
{
    return std::vector{
        // clang-format off
        // 3x3 s1 -> channel-first floor (v2_core_nchw)
        TestCase{{dt, miopenTensorNCHW, {2, 8, 16, 16}}, {dt, miopenTensorNCHW, {8, 1, 3, 3}},
                 dt, {{1, 1}, {1, 1}, {1, 1}, 8}},
        // 5x5 s2 -> channel-first floor
        TestCase{{dt, miopenTensorNCHW, {2, 8, 16, 16}}, {dt, miopenTensorNCHW, {8, 1, 5, 5}},
                 dt, {{2, 2}, {2, 2}, {1, 1}, 8}},
        // 7x7 s1 -> channel-first floor (halo variants decline in NCHW)
        TestCase{{dt, miopenTensorNCHW, {2, 8, 16, 16}}, {dt, miopenTensorNCHW, {8, 1, 7, 7}},
                 dt, {{3, 3}, {1, 1}, {1, 1}, 8}},
        // non-square kernel s1 -> channel-first floor
        TestCase{{dt, miopenTensorNCHW, {1, 12, 20, 24}}, {dt, miopenTensorNCHW, {12, 1, 3, 5}},
                 dt, {{1, 2}, {1, 1}, {1, 1}, 12}},
        // clang-format on
    };
}

// -------- 3D: channel-last (NDHWC) — floor + 3D LDS variant -----------------
auto GetConvTestCasesNDHWC3D(miopenDataType_t dt)
{
    return std::vector{
        // clang-format off
        // 3x3x3 s1 -> 3D LDS variant (v3b_lds_core_ndhwc)
        TestCase{{dt, miopenTensorNDHWC, {1, 8, 8, 16, 16}}, {dt, miopenTensorNDHWC, {8, 1, 3, 3, 3}},
                 dt, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8}},
        // 3x3x3 s2 -> 3D floor (v2_wstrip_core_ndhwc; LDS variant requires s==1)
        TestCase{{dt, miopenTensorNDHWC, {1, 8, 8, 16, 16}}, {dt, miopenTensorNDHWC, {8, 1, 3, 3, 3}},
                 dt, {{1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 8}},
        // clang-format on
    };
}

// -------- 3D: channel-first (NCDHW) — the new capability, floor path --------
auto GetConvTestCasesNCDHW3D(miopenDataType_t dt)
{
    return std::vector{
        // clang-format off
        // 3x3x3 s1 -> channel-first 3D floor (v2_core_ncdhw); LDS declines in NCDHW
        TestCase{{dt, miopenTensorNCDHW, {1, 8, 8, 16, 16}}, {dt, miopenTensorNCDHW, {8, 1, 3, 3, 3}},
                 dt, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8}},
        // 3x3x3 s2 -> channel-first 3D floor
        TestCase{{dt, miopenTensorNCDHW, {1, 8, 8, 16, 16}}, {dt, miopenTensorNCDHW, {8, 1, 3, 3, 3}},
                 dt, {{1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 8}},
        // clang-format on
    };
}

// The union used for the Full instantiation (all layouts, both dimensionalities).
auto GetConvFullTestCases(miopenDataType_t dt)
{
    auto cases = GetConvTestCasesNHWC2D(dt);
    for(auto& v :
        {GetConvTestCasesNCHW2D(dt), GetConvTestCasesNDHWC3D(dt), GetConvTestCasesNCDHW3D(dt)})
        cases.insert(cases.end(), v.begin(), v.end());
    return cases;
}

// One representative per (layout, dim) for the Smoke instantiation.
auto GetConvSmokeTestCases(miopenDataType_t dt)
{
    return std::vector{
        GetConvTestCasesNHWC2D(dt)[0],  // NHWC 2D (microtile)
        GetConvTestCasesNCHW2D(dt)[0],  // NCHW 2D floor (new)
        GetConvTestCasesNDHWC3D(dt)[0], // NDHWC 3D (LDS)
        GetConvTestCasesNCDHW3D(dt)[0], // NCDHW 3D floor (new)
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        // RDNA only: gfx11xx (RDNA3), gfx115x (RDNA3.5), gfx120x (RDNA4). Not
        // gfx125x (gfx1250 is CDNA5 despite the gfx12 prefix). gfx942/CDNA3
        // is scaffold-only (every CDNA3 row is wip), so the solver is cleanly
        // not-applicable there. gfx1103 (a small RDNA3 APU) IS supported — the
        // VALU cores are portable HIP with no gfx1103-specific issue, so the
        // solver admits it and the dev-applicability check expects it here too.
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx110X | Gpu::gfx115X |
                                                              Gpu::gfx120X);
        p.Tunable(5); // exercise the index-into-table search over the valid subset
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverDepthwiseDirectFwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverDepthwiseDirectFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using CPU_UnitTestConvSolverDepthwiseDirectDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverDepthwiseDirectFwd_FP16, ConvDepthwiseDirect)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseDirect{});
};

TEST_P(GPU_UnitTestConvSolverDepthwiseDirectFwd_BFP16, ConvDepthwiseDirect)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseDirect{});
};

TEST_P(CPU_UnitTestConvSolverDepthwiseDirectDevApplicabilityFwd_NONE, ConvDepthwiseDirect)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseDirect{});
};

// Smoke tests (fp16 + bf16): one case per (layout, dim).
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverDepthwiseDirectFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverDepthwiseDirectFwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

// Full tests (fp16 + bf16): every variant, both layouts, 2D and 3D.
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverDepthwiseDirectFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverDepthwiseDirectFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

// Device-applicability test: detects any accidental expansion of the solver's
// device coverage (runs on CPU; no kernel launch).
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDepthwiseDirectDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

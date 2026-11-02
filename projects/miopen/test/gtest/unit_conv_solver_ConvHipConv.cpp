// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the ConvHipConv solver (hipconv-backed convolution).
//
// Restrictions enforced by ConvHipConv::IsApplicable:
//   - 2D convolution, or a 3D convolution that reduces to one
//   - packed tensors
//   - fp16, bf16, or fp32 with tf32 compute enabled
//   - NHWC/NDHWC, or 2D NCHW (served by transposing through NHWC scratch); 3D NCDHW is
//     not applicable, MakeTransposePlan() being 4D
//   - architectures recognised by hipconv (gfx950, gfx1250)
//   - the hipconv library must have a valid kernel for the (params, direction) tuple

#include "unit_conv_solver.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

namespace {

using TestCase = miopen::unit_tests::ConvTestCase;

// Small representative cases (one per channels-per-group family) for smoke runs.
//
// Run in both layouts: NHWC reaches hipconv directly, NCHW exercises the solver's
// NCHW<->NHWC staging (input/weight transposes in, result transposed back out).
//
// tf32_compute selects tf32 MFMA compute for an fp32 problem: ConvHipConv's
// IsSupportedProblem() only allows fp32 via UseTF32(), plain pedantic fp32 is not
// applicable.
auto GetConvSmokeTestCases(miopenDataType_t datatype,
                           miopenTensorLayout_t layout,
                           bool tf32_compute = false)
{
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64,  4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16, false, tf32_compute}}, // 4c
        TestCase{{datatype, layout, {4, 16, 8, 1}}, {datatype, layout, {16,  8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32_compute}}, // 8c
        TestCase{{datatype, layout, {4, 32, 8, 1}}, {datatype, layout, {32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32_compute}}, // 16c
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32_compute}}, // 32c
        // clang-format on
    };
}

// The same cases at depth 2, with depth left unconvolved so each folds into the batch.
//
// NDHWC only: 3D reaches hipconv directly, and the NCDHW staging path does not exist.
auto GetConv3dSmokeTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNDHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 2, 8, 1}}, {datatype, layout, {64,  4, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1}, 16, false, tf32_compute}}, // 4c
        TestCase{{datatype, layout, {4, 16, 2, 8, 1}}, {datatype, layout, {16,  8, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 8c
        TestCase{{datatype, layout, {4, 32, 2, 8, 1}}, {datatype, layout, {32, 16, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 16c
        TestCase{{datatype, layout, {4, 64, 2, 8, 1}}, {datatype, layout, {64, 32, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 32c
        // clang-format on
    };
}

// The same cases with the depth dimension convolved instead of the spatial ones.
//
// Distinct from the above in where the filter's non-unit extent sits: here it is
// z, which a solver reading only y and x drops. No tf32 instantiation, because
// hipconv has no kernel for these at fp32. gfx950 only: they unfold to vertical filters.
auto GetConv3dDepthSmokeTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNDHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 8, 8, 1}}, {datatype, layout, {64,  4, 3, 1, 1}}, datatype, {{1, 0, 0}, {1, 1, 1}, {1, 1, 1}, 16, false, tf32_compute}}, // 4c
        TestCase{{datatype, layout, {4, 16, 8, 8, 1}}, {datatype, layout, {16,  8, 3, 1, 1}}, datatype, {{1, 0, 0}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 8c
        TestCase{{datatype, layout, {4, 32, 8, 8, 1}}, {datatype, layout, {32, 16, 3, 1, 1}}, datatype, {{1, 0, 0}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 16c
        TestCase{{datatype, layout, {4, 64, 8, 8, 1}}, {datatype, layout, {64, 32, 3, 1, 1}}, datatype, {{1, 0, 0}, {1, 1, 1}, {1, 1, 1},  2, false, tf32_compute}}, // 32c
        // clang-format on
    };
}

// Dense wgrad shapes whose channel count leaves a partial tile, which is where the
// direct_wgrad epilogue's channel guard splits a wave. The defect these cover (ROCM-31508)
// appeared only on non-default perf configs, which GetTestParams()'s Tunable(5) sweeps, and
// only on channel counts that are not a multiple of the 32-wide wave tile.
auto GetConvWrwDenseTestCases(miopenDataType_t datatype, miopenTensorLayout_t layout)
{
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {1,  40,  8, 32}}, {datatype, layout, { 70,  40, 5, 5}}, datatype, {{0, 0}, {1, 1}, {1, 1}}}, // C(40)
        TestCase{{datatype, layout, {9, 100, 12, 32}}, {datatype, layout, {128, 100, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}}}, // C(100)
        TestCase{{datatype, layout, {6, 162,  8,  8}}, {datatype, layout, {128, 162, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}}}, // C(162)
        // clang-format on
    };
}

// The lists below probe each hipconv kernel, NHWC only: the NCHW staging does not depend on
// the kernel, and the lists above cover it.

// Dense layers, one per filter shape the direct kernels specialize on.
//
// gfx950 serves them with direct_l1 or direct, and direct_wgrad; gfx1250 with direct and
// direct_wgrad.
auto GetConvDenseTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {9,  10, 16, 32}}, {datatype, layout, { 64, 10, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}}, // C(10)
        TestCase{{datatype, layout, {1,  40,  8, 32}}, {datatype, layout, { 70, 40, 2, 2}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}}, // 2x2
        TestCase{{datatype, layout, {7,  64,  8,  8}}, {datatype, layout, {128, 64, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}}, // 3x3
        TestCase{{datatype, layout, {4,  64,  8,  8}}, {datatype, layout, { 64, 32, 4, 4}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2, false, tf32_compute}}, // 4x4
        TestCase{{datatype, layout, {4,  64,  8,  8}}, {datatype, layout, { 64, 32, 5, 5}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2, false, tf32_compute}}, // 5x5
        TestCase{{datatype, layout, {2, 256,  4,  8}}, {datatype, layout, {256, 32, 5, 5}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 8, false, tf32_compute}}, // 5x5, 8 groups
        // clang-format on
    };
}

// Vertical 3x1 filters: gfx950 serves every direction (direct_l1, direct_wgrad), gfx1250 Wrw
// only (direct_wgrad).
auto GetConvDenseVerticalTestCases(miopenDataType_t datatype)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {1,  40, 8, 32}}, {datatype, layout, { 70, 40, 3, 1}}, datatype, {{1, 1}, {1, 1}, {1, 1}}},    // C(40)
        TestCase{{datatype, layout, {2, 128, 8, 16}}, {datatype, layout, {128, 32, 3, 1}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 4}}, // 4 groups
        // clang-format on
    };
}

// Patch embedding: stride equal to the filter and no padding (patch_embed, gfx950 only).
auto GetConvPatchEmbedTestCases(miopenDataType_t datatype)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {2, 3,  32,  32}}, {datatype, layout, {64, 3, 16, 16}}, datatype, {{0, 0}, {16, 16}, {1, 1}}}, // kw*c a multiple of 8
        TestCase{{datatype, layout, {1, 5,  22,  22}}, {datatype, layout, {32, 5, 11, 11}}, datatype, {{0, 0}, {11, 11}, {1, 1}}}, // kw*c odd
        TestCase{{datatype, layout, {2, 3,   8,  16}}, {datatype, layout, {64, 3,  2,  4}}, datatype, {{0, 0}, { 2,  4}, {1, 1}}}, // non-square patch
        TestCase{{datatype, layout, {4, 3,  64,  64}}, {datatype, layout, {61, 3, 16, 16}}, datatype, {{0, 0}, {16, 16}, {1, 1}}}, // K(61)
        TestCase{{datatype, layout, {5, 3, 112, 112}}, {datatype, layout, {96, 3, 16, 16}}, datatype, {{0, 0}, {16, 16}, {1, 1}}}, // wgrad split across workgroups
        // clang-format on
    };
}

// Odd channel counts, which the gfx950 16-bit kernels reject.
//
// gfx950 serves them in tf32 only (direct_l1, direct_wgrad); gfx1250 serves them in 16-bit
// (direct).
auto GetConvOddChannelTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {2, 19, 32, 32}}, {datatype, layout, { 65, 19, 4, 4}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 1, false, tf32_compute}}, // C(19)
        TestCase{{datatype, layout, {2, 65, 32, 32}}, {datatype, layout, { 19, 65, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 1, false, tf32_compute}}, // K(19)
        TestCase{{datatype, layout, {2, 12, 32, 32}}, {datatype, layout, { 16,  3, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 4, false, tf32_compute}}, // 3 channels per group
        // clang-format on
    };
}

// 1x1, which gfx1250 serves (direct) and no gfx950 kernel in MIOpen's hipconv build does.
auto GetConvPointwiseTestCases(miopenDataType_t datatype)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4,  64, 32, 32}}, {datatype, layout, {  3,  64, 1, 1}}, datatype, {{0, 0}, {1, 1}, {1, 1}}}, // K(3)
        TestCase{{datatype, layout, {4, 127, 32, 32}}, {datatype, layout, {129, 127, 1, 1}}, datatype, {{0, 0}, {1, 1}, {1, 1}}}, // C(127), K(129)
        // clang-format on
    };
}

// Grouped 3x3 at stride 1, one layer per channels-per-group width.
//
// The direct kernels serve these too, so the suites use GetTestParamsTopRanked().
auto GetConvGroupedTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4,  64, 8,  7}}, {datatype, layout, { 64,  4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16, false, tf32_compute}}, // 4c
        TestCase{{datatype, layout, {4,  64, 3,  5}}, {datatype, layout, { 64,  8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  8, false, tf32_compute}}, // 8c
        TestCase{{datatype, layout, {2, 256, 8, 16}}, {datatype, layout, {256, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16, false, tf32_compute}}, // 16c
        TestCase{{datatype, layout, {2, 256, 4,  8}}, {datatype, layout, {256, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  8, false, tf32_compute}}, // 32c
        // clang-format on
    };
}

// Grouped 3x3 at stride 2, which no kernel serves in wgrad.
auto GetConvGroupedStride2TestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4,  64,  8,  9}}, {datatype, layout, { 64,  4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16, false, tf32_compute}}, // 4c
        TestCase{{datatype, layout, {4, 128, 16, 32}}, {datatype, layout, {128,  4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32, false, tf32_compute}}, // 4c, 32 groups
        TestCase{{datatype, layout, {4,  32,  8,  9}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2, false, tf32_compute}}, // 16c
        TestCase{{datatype, layout, {5,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2, false, tf32_compute}}, // 16c, odd batch
        TestCase{{datatype, layout, {4,  64,  8,  9}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2, false, tf32_compute}}, // 32c
        TestCase{{datatype, layout, {4, 128, 16, 32}}, {datatype, layout, {128, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  4, false, tf32_compute}}, // 32c, 4 groups
        // clang-format on
    };
}

// Grouped 3x3 at stride 2 with 8 channels per group, which gfx950 serves in fprop only.
auto GetConvGrouped8cStride2TestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 16,  8,  7}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2, false, tf32_compute}}, // 2 groups
        TestCase{{datatype, layout, {4, 64, 16, 64}}, {datatype, layout, {64, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 8, false, tf32_compute}}, // 8 groups
        // clang-format on
    };
}

// Depthwise, one layer per filter size, two of them at stride 2.
//
// depthwise_1d_toeplitz and depthwise_wgrad_hankel on both architectures. No kernel serves a
// tf32 depthwise wgrad at every size here, so there is no TF32 Wrw suite.
auto GetConvDepthwiseTestCases(miopenDataType_t datatype, bool tf32_compute = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {2, 64, 17, 19}}, {datatype, layout, {64, 1,  3,  3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 64, false, tf32_compute}}, // 3x3
        TestCase{{datatype, layout, {3, 65, 19, 37}}, {datatype, layout, {65, 1,  5,  5}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 65, false, tf32_compute}}, // 5x5
        TestCase{{datatype, layout, {7, 32, 21, 33}}, {datatype, layout, {32, 1,  7,  7}}, datatype, {{0, 3}, {1, 1}, {1, 1}, 32, false, tf32_compute}}, // 7x7
        TestCase{{datatype, layout, {2, 64, 17, 19}}, {datatype, layout, {64, 1, 11, 11}}, datatype, {{5, 5}, {1, 1}, {1, 1}, 64, false, tf32_compute}}, // 11x11
        TestCase{{datatype, layout, {4, 20, 21, 21}}, {datatype, layout, {20, 1,  5,  5}}, datatype, {{2, 2}, {2, 2}, {1, 1}, 20, false, tf32_compute}}, // 5x5, stride 2
        TestCase{{datatype, layout, {2, 64, 17, 19}}, {datatype, layout, {64, 1,  9,  9}}, datatype, {{4, 4}, {2, 2}, {1, 1}, 64, false, tf32_compute}}, // 9x9, stride 2
        // clang-format on
    };
}

auto MakeTestParams(Gpu supported_gpus, std::size_t tuning_iterations)
{
    auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
    p.Tunable(tuning_iterations);
    return p;
}

// gfx1250 has no CI test runner yet, so it skips there until one exists.
const auto& GetTestParams()
{
    static const auto params = MakeTestParams(Gpu::gfx950 | Gpu::gfx125X, 5);
    return params;
}

const auto& GetTestParamsGfx950()
{
    static const auto params = MakeTestParams(Gpu::gfx950, 5);
    return params;
}

const auto& GetTestParamsGfx125X()
{
    static const auto params = MakeTestParams(Gpu::gfx125X, 5);
    return params;
}

// Verifies hipconv's top-ranked config, which is the solver's default.
//
// GenericSearch samples configs at random, so on a layer several kernels serve, Tunable(5)
// verifies whichever kernel's sample is fastest. Tunable(0) samples none and runs the default.
const auto& GetTestParamsTopRanked()
{
    static const auto params = MakeTestParams(Gpu::gfx950 | Gpu::gfx125X, 0);
    return params;
}

// MIOpen enables tf32 compute on gfx942 and gfx95x only (IsTF32Supported), so the tf32 suites
// run on gfx950 alone, although hipconv's gfx1250 kernels also serve tf32.
const auto& GetTestParamsTF32()
{
    static const auto params = MakeTestParams(Gpu::gfx950, 5);
    return params;
}

const auto& GetTestParamsTF32TopRanked()
{
    static const auto params = MakeTestParams(Gpu::gfx950, 0);
    return params;
}

} // namespace

using GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16 = GPU_UnitTestConvSolverWrw_FP16;

using GPU_UnitTestConvSolverConvHipConvFwdNchw_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvHipConvBwdNchw_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverConvHipConvWrwNchw_FP16 = GPU_UnitTestConvSolverWrw_FP16;

using GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;

using GPU_UnitTestConvSolverConvHipConvFwdNchw_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvBwdNchw_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvWrwNchw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;

using GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32 = GPU_UnitTestConvSolverFwd_TF32;
using GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32 = GPU_UnitTestConvSolverBwd_TF32;
using GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32 = GPU_UnitTestConvSolverWrw_TF32;

using GPU_UnitTestConvSolverConvHipConvFwdNchw_TF32 = GPU_UnitTestConvSolverFwd_TF32;
using GPU_UnitTestConvSolverConvHipConvBwdNchw_TF32 = GPU_UnitTestConvSolverBwd_TF32;
using GPU_UnitTestConvSolverConvHipConvWrwNchw_TF32 = GPU_UnitTestConvSolverWrw_TF32;

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNchw_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNchw_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNchw_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNchw_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNchw_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNchw_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwdNchw_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwdNchw_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrwNchw_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNHWC))));

// NCHW: same shapes through the solver's NCHW<->NHWC staging path.

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNchw_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNchw_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNchw_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenHalf, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNchw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNchw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNchw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvFwdNchw_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNCHW, true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvBwdNchw_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNCHW, true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvWrwNchw_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(
                                              miopenFloat, miopenTensorNCHW, true))));

// 3D, NDHWC only: the NCDHW staging path does not exist, so these reuse the Nhwc suites.

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3dDepth,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dDepthSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseWrw,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvWrwDenseTestCases(miopenHalf, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseWrw,
    GPU_UnitTestConvSolverConvHipConvWrwNchw_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvWrwDenseTestCases(miopenHalf, miopenTensorNCHW))));

INSTANTIATE_TEST_SUITE_P(SmokeDenseWrw,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvWrwDenseTestCases(
                                              miopenBFloat16, miopenTensorNHWC))));

INSTANTIATE_TEST_SUITE_P(SmokeDenseWrw,
                         GPU_UnitTestConvSolverConvHipConvWrwNchw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvWrwDenseTestCases(
                                              miopenBFloat16, miopenTensorNCHW))));

// Kernel probes, NHWC only.

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDense,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDense,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDense,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(SmokeDense,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDenseTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDenseVertical,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDenseVerticalTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokePatchEmbed,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx950()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPatchEmbedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeOddChannels,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvOddChannelTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeOddChannels,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvOddChannelTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeOddChannels,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvOddChannelTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeOddChannels,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvOddChannelTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeOddChannels,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvOddChannelTestCases(miopenFloat,
                                                                                       true))));

INSTANTIATE_TEST_SUITE_P(SmokeOddChannels,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvOddChannelTestCases(miopenFloat,
                                                                                       true))));

INSTANTIATE_TEST_SUITE_P(SmokeOddChannels,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvOddChannelTestCases(miopenFloat,
                                                                                       true))));

INSTANTIATE_TEST_SUITE_P(
    SmokePointwise,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPointwiseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokePointwise,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPointwiseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokePointwise,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPointwiseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokePointwise,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvPointwiseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParamsTopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsTopRanked()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParamsTopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsTopRanked()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
                         testing::Combine(testing::Values(GetTestParamsTopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsTopRanked()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32TopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32TopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeGrouped,
                         GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32TopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvGroupedTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
    testing::Combine(testing::Values(GetTestParamsTF32()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenFloat, true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGroupedStride2,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
    testing::Combine(testing::Values(GetTestParamsTF32()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGroupedStride2TestCases(miopenFloat, true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped8cStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGrouped8cStride2TestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped8cStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGrouped8cStride2TestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped8cStride2,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGrouped8cStride2TestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped8cStride2,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParamsGfx125X()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGrouped8cStride2TestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeGrouped8cStride2,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
    testing::Combine(testing::Values(GetTestParamsTF32()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvGrouped8cStride2TestCases(miopenFloat, true))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeDepthwise,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvDepthwiseTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeDepthwise,
                         GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32TopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDepthwiseTestCases(miopenFloat,
                                                                                      true))));

INSTANTIATE_TEST_SUITE_P(SmokeDepthwise,
                         GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
                         testing::Combine(testing::Values(GetTestParamsTF32TopRanked()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvDepthwiseTestCases(miopenFloat,
                                                                                      true))));

#endif // MIOPEN_USE_HIPCONV

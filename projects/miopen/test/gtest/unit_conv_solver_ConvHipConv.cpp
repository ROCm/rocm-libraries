// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the ConvHipConv solver (hipconv-backed grouped 3x3 conv).
//
// Restrictions enforced by ConvHipConv::IsApplicable:
//   - 2D convolution only
//   - fp16 or bf16
//   - NHWC, or NCHW with packed tensors (served by transposing through NHWC scratch)
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

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx950 | Gpu::gfx125X;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.Tunable(5);
        return p;
    }();
    return params;
}

// hipconv's tf32 kernels are cdna4-only (see src/hipconv/src/arch/cdna4/grouped/
// *_wgrad_tf32.cpp and the tf32 branches in grouped_{4,8,16,32}c.cpp); cdna5 (gfx1250)
// has no tf32 kernel family at all, so this suite is gfx950-only, unlike the fp16/bf16
// suites above which also run on gfx1250.
const auto& GetTestParamsTF32()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx950);
        p.Tunable(5);
        return p;
    }();
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

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNhwc_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNhwc_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNhwc_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNHWC, true))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwdNchw_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNCHW, true))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwdNchw_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNCHW, true))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrwNchw_TF32,
    testing::Combine(
        testing::Values(GetTestParamsTF32()),
        testing::Values(miopenConvolutionAlgoDirect),
        testing::ValuesIn(GetConvSmokeTestCases(miopenFloat, miopenTensorNCHW, true))));

#endif // MIOPEN_USE_HIPCONV

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the ConvHipConv solver (hipconv-backed grouped 3x3 conv).
//
// Restrictions enforced by ConvHipConv::IsApplicable:
//   - 2D convolution, or a 3D convolution that reduces to one
//   - packed tensors
//   - fp16, bf16, or fp32 with tf32 compute enabled
//   - architectures recognised by hipconv (gfx950, gfx1250)
//   - the hipconv library must have a valid kernel for the (params, direction) tuple

#include "unit_conv_solver.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

namespace {

using TestCase = miopen::unit_tests::ConvTestCase;

// Small representative cases (one per channels-per-group family) for smoke runs.
auto GetConvSmokeTestCases(miopenDataType_t datatype, bool tf32 = false)
{
    constexpr auto layout = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64,  4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16, false, tf32}}, // 4c
        TestCase{{datatype, layout, {4, 16, 8, 1}}, {datatype, layout, {16,  8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32}}, // 8c
        TestCase{{datatype, layout, {4, 32, 8, 1}}, {datatype, layout, {32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32}}, // 16c
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2, false, tf32}}, // 32c
        // clang-format on
    };
}

// The same cases at depth 2, with depth left unconvolved so each folds into the batch.
auto GetConv3dSmokeTestCases(miopenDataType_t datatype, bool tf32 = false)
{
    constexpr auto layout = miopenTensorNDHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 2, 8, 1}}, {datatype, layout, {64,  4, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1}, 16, false, tf32}}, // 4c
        TestCase{{datatype, layout, {4, 16, 2, 8, 1}}, {datatype, layout, {16,  8, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32}}, // 8c
        TestCase{{datatype, layout, {4, 32, 2, 8, 1}}, {datatype, layout, {32, 16, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32}}, // 16c
        TestCase{{datatype, layout, {4, 64, 2, 8, 1}}, {datatype, layout, {64, 32, 1, 3, 3}}, datatype, {{0, 1, 1}, {1, 1, 1}, {1, 1, 1},  2, false, tf32}}, // 32c
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        // gfx1250 has no CI test runner yet, so it skips there until one exists.
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx950 | Gpu::gfx125X);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverConvHipConvFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvHipConvBwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverConvHipConvWrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;

using GPU_UnitTestConvSolverConvHipConvFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvBwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverConvHipConvWrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;

using GPU_UnitTestConvSolverConvHipConvFwd_TF32 = GPU_UnitTestConvSolverFwd_TF32;
using GPU_UnitTestConvSolverConvHipConvBwd_TF32 = GPU_UnitTestConvSolverBwd_TF32;
using GPU_UnitTestConvSolverConvHipConvWrw_TF32 = GPU_UnitTestConvSolverWrw_TF32;

TEST_P(GPU_UnitTestConvSolverConvHipConvFwd_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwd_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrw_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwd_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwd_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrw_BFP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvFwd_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwd_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrw_TF32, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvFwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvBwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvHipConvWrw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvFwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvBwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    SmokeConv3d,
    GPU_UnitTestConvSolverConvHipConvWrw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConv3dSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvFwd_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvBwd_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvWrw_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat,
                                                                                  true))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvFwd_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvBwd_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

INSTANTIATE_TEST_SUITE_P(SmokeConv3d,
                         GPU_UnitTestConvSolverConvHipConvWrw_TF32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConv3dSmokeTestCases(miopenFloat,
                                                                                    true))));

#endif // MIOPEN_USE_HIPCONV

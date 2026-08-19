// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {
// ConvDepthwiseBwdData2D computes the input gradient for stride-1 depthwise convs by
// reusing the forward CK kernel with a 180-degree-rotated filter, so all cases here are
// stride-1 depthwise (group count == channels) with an odd, symmetric filter.

auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    // Single small 7x7 stride-1 pad-3 depthwise case (convnext-style).
    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 7, 7}},
                 {datatype, miopenTensorNCHW, {2, 1, 7, 7}},
                 datatype, {{3, 3}, {1, 1}, {1, 1}, 2}},
        // clang-format on
    };
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    // Cover the stride-1 dgrad instances: 7x7 pad-3 (tiles 7/14/28/56),
    // 3x3 pad-1 (tiles 7/14/56), 5x5 pad-2 (tiles 7/14/28), group==channels.
    return std::vector{
        // clang-format off
        // 7x7 stride-1 pad-3 (convnext depthwise stages)
        TestCase{{datatype, miopenTensorNCHW, {32, 1,  7,  7}},
                 {datatype, miopenTensorNCHW, {1, 1, 7, 7}}, datatype, {{3, 3}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 14, 14}},
                 {datatype, miopenTensorNCHW, {2, 1, 7, 7}}, datatype, {{3, 3}, {1, 1}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 28, 28}},
                 {datatype, miopenTensorNCHW, {2, 1, 7, 7}}, datatype, {{3, 3}, {1, 1}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 56, 56}},
                 {datatype, miopenTensorNCHW, {2, 1, 7, 7}}, datatype, {{3, 3}, {1, 1}, {1, 1}, 2}},
        // 3x3 stride-1 pad-1
        TestCase{{datatype, miopenTensorNCHW, {32, 1,  7,  7}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {32, 3, 14, 14}},
                 {datatype, miopenTensorNCHW, {3, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 3}},
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 56, 56}},
                 {datatype, miopenTensorNCHW, {2, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}},
        // 5x5 stride-1 pad-2
        TestCase{{datatype, miopenTensorNCHW, {32, 1,  7,  7}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 14, 14}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {32, 4, 28, 28}},
                 {datatype, miopenTensorNCHW, {4, 1, 5, 5}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 4}},
        // clang-format on
    };
}

auto GetTestParams()
{
// Reuses the forward CK depthwise kernel: needs 64-lane wavefronts + the CK dynamic library.
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);
    params.UsesCKDynamicLib();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverConvDepthwiseBwdData2D_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverConvDepthwiseBwdData2D_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using CPU_UnitTestConvSolverConvDepthwiseBwdData2DDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDepthwiseBwdData2D_FP16, ConvDepthwiseBwdData2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseBwdData2D{});
};

TEST_P(GPU_UnitTestConvSolverConvDepthwiseBwdData2D_BFP16, ConvDepthwiseBwdData2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseBwdData2D{});
};

TEST_P(CPU_UnitTestConvSolverConvDepthwiseBwdData2DDevApplicability_NONE, ConvDepthwiseBwdData2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseBwdData2D{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDepthwiseBwdData2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverConvDepthwiseBwdData2D_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvDepthwiseBwdData2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvDepthwiseBwdData2D_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDepthwiseBwdData2DDevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

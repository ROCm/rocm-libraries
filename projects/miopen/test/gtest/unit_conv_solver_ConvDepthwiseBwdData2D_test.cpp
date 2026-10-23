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
    // 3x3 pad-1 (tiles 7/14/28/56), 5x5 pad-2 (tiles 7/14/28), group==channels.
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
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 28, 28}},
                 {datatype, miopenTensorNCHW, {2, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}},
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

auto GetConvWave32OnlyTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    // 3x3 stride-1 pad-1 tile-112 (efficientnet/mobilenet first depthwise stage). This tile
    // exists only in the wave32 instance set, so the case is restricted to RDNA. The wave64
    // factory has no matching instance and the solver would decline the problem there.
    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNCHW, {32, 2, 112, 112}},
                 {datatype, miopenTensorNCHW, {2, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}},
        // clang-format on
    };
}

auto GetTestParams()
{
// Reuses the forward CK depthwise kernel and needs the CK dynamic library. Runs on CDNA (64-lane)
// and RDNA (32-lane) wavefronts; the matching instance set is selected per arch at build time.
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx110X |
                           Gpu::gfx115X | Gpu::gfx120X | Gpu::gfx125X;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);
    params.UsesCKDynamicLib();
    return params;
}

auto GetWave32TestParams()
{
// Same as GetTestParams(), minus the CDNA devices: the cases using these params need an
// instance that only the wave32 set provides.
#if MIOPEN_BACKEND_HIP
    Gpu supportedDevices = Gpu::gfx110X | Gpu::gfx115X | Gpu::gfx120X | Gpu::gfx125X;
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

// Wave32-only tests (RDNA). Kept apart from the shared cases above because the wave64
// factory has no 112x112 instance.
INSTANTIATE_TEST_SUITE_P(
    FullWave32,
    GPU_UnitTestConvSolverConvDepthwiseBwdData2D_FP16,
    testing::Combine(testing::Values(GetWave32TestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvWave32OnlyTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    FullWave32,
    GPU_UnitTestConvSolverConvDepthwiseBwdData2D_BFP16,
    testing::Combine(testing::Values(GetWave32TestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvWave32OnlyTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDepthwiseBwdData2DDevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

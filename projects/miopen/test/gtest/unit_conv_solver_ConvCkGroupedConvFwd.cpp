// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {
auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    // Smoke test using a single case with batchsize of 32, 1 channel & group, (2,2) padding,
    // (5,5) filter and (7,7) spatial dims.
    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNCHW, {32, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        // clang-format on
    };
}

auto GetConvDeterministicTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {32, 1, 7, 7}},
                 {datatype, miopenTensorNHWC, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1, true}},
        // clang-format on
    };
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // For every entry in DeviceConvFwdFactory, test with a group count of 1 and with a group count
    // greater than 1
    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 2, 7, 7}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 14, 14}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 3, 14, 14}},
                 {datatype, miopenTensorNCHW, {3, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 3}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 28, 28}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 4, 28, 28}},
                 {datatype, miopenTensorNCHW, {4, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 4}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 14, 14}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 2, 14, 14}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 28, 28}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 2, 28, 28}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 56, 56}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 2, 56, 56}},
                 {datatype, miopenTensorNCHW, {2, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 2}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 3, 7, 7}},
                 {datatype, miopenTensorNCHW, {3, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 3}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 14, 14}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 4, 14, 14}},
                 {datatype, miopenTensorNCHW, {4, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 4}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 56, 56}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 5, 56, 56}},
                 {datatype, miopenTensorNCHW, {5, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 5}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 112, 112}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 2, 112, 112}},
                 {datatype, miopenTensorNCHW, {2, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, 
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 28, 28}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {2, 2}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 6, 28, 28}},
                 {datatype, miopenTensorNCHW, {6, 1, 3, 3}},
                 datatype, {{1, 1}, {2, 2}, {1, 1}, 6}},
        TestCase{{datatype, miopenTensorNCHW, {64, 1, 112, 112}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {2, 2}, {1, 1}, 1}},
        TestCase{{datatype, miopenTensorNCHW, {64, 3, 112, 112}},
                 {datatype, miopenTensorNCHW, {3, 1, 3, 3}},
                 datatype, {{1, 1}, {2, 2}, {1, 1}, 3}},
        // clang-format on
    };
}

auto GetTestParams(miopenDataType_t /*datatype*/)
{
// Solution requires 64-lane wavefronts and depends on the CK dynamic library
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
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

using GPU_UnitTestConvSolverConvDepthwiseFwd2D_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using CPU_UnitTestConvSolverConvDepthwiseFwd2DDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverConvDepthwiseFwd2DeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDepthwiseFwd2D_FP16, ConvDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseFwd2D{});
};

TEST_P(CPU_UnitTestConvSolverConvDepthwiseFwd2DDevApplicability_FP16, ConvDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseFwd2D{});
};

TEST_P(CPU_UnitTestConvSolverConvDepthwiseFwd2DeterministicApplicability_NONE, ConvDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseFwd2D{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDepthwiseFwd2DDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverConvDepthwiseFwd2DeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None),
                     testing::Values(GetConvDeterministicTestCases(miopenHalf)[0])));

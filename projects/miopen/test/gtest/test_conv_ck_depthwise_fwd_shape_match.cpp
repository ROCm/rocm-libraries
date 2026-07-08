// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the CK depthwise forward shape-match guard.
//
// Each CK instance in DeviceConvFwdFactory is hard-specialized at compile time for a specific
// filter size, stride, padding and dilation. CK's IsSupportedArgument only validates
// tiling/divisibility and does NOT reject a shape-mismatched instance. Before the fix,
// FillValidKernels / CheckCKApplicability / CheckIsArgSupported could therefore accept (and the
// tuner could record) an instance whose compile-time shape did not match the problem -- e.g. a
// FilterSize=5 instance for a 3x3 problem -- producing a perf config that later failed to build.
//
// InstanceShapeMatchesProblem() now gates applicability on an exact shape match, and
// ConvDepthwiseFwd2D::GetSolution() rejects a kernel_id absent from this build's factory instead
// of aborting. This test drives ConvDepthwiseFwd2D end-to-end (tune -> build -> numeric validate)
// across several DISTINCT filter shapes; a wrong-shape selection would fail to build or produce
// incorrect results, and the CPU applicability case guards against silent applicability expansion.

#include "unit_conv_solver.hpp"

namespace {

// Test cases spanning multiple distinct compile-time shapes present in DeviceConvFwdFactory
// (3x3 stride-1 pad-1, 5x5 stride-1 pad-2, 5x5 stride-2 pad-2). Each requires the solver to pick
// the instance whose specialization matches the problem exactly.
auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // 3x3, stride 1, pad 1 -- must NOT match any FilterSize=5 instance
        TestCase{{datatype, miopenTensorNCHW, {32, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}, 1}},
        // 5x5, stride 1, pad 2 -- must NOT match any FilterSize=3 instance
        TestCase{{datatype, miopenTensorNCHW, {32, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {1, 1}, {1, 1}, 1}},
        // 5x5, stride 2, pad 2 -- distinguishes stride within the same filter size
        TestCase{{datatype, miopenTensorNCHW, {32, 1, 14, 14}},
                 {datatype, miopenTensorNCHW, {1, 1, 5, 5}},
                 datatype, {{2, 2}, {2, 2}, {1, 1}, 1}},
        // clang-format on
    };
}

// A shape that matches NO compiled instance: 3x3 with pad 0 (all FilterSize=3 instances are
// specialized for pad 1). The solver must report this as not-applicable rather than selecting a
// shape-mismatched instance. Used for the device-applicability guard.
auto GetUnsupportedShapeTestCase(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNCHW, {32, 1, 7, 7}},
                 {datatype, miopenTensorNCHW, {1, 1, 3, 3}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}, 1}},
        // clang-format on
    };
}

auto GetTestParams()
{
// Solution requires 64-lane wavefronts and depends on the CK dynamic library.
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

// For the negative applicability case: supported_devs = None (so the framework expects
// IsApplicable() == false everywhere) while UsesCKDynamicLib() restricts the check to the real
// current device with the CK library loaded -- so the shape-match guard is actually exercised and
// must reject the shape, rather than being skipped on a mock handle with no CK library.
auto GetNegativeApplicabilityParams()
{
    auto params = miopen::unit_tests::UnitTestConvSolverParams(Gpu::None);
    params.UsesCKDynamicLib();

    return params;
}

} // namespace

using GPU_UnitTestConvCkDepthwiseFwdShapeMatch_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using CPU_UnitTestConvCkDepthwiseFwdShapeMatchDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvCkDepthwiseFwdShapeMatch_FP16, ConvDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseFwd2D{});
};

TEST_P(CPU_UnitTestConvCkDepthwiseFwdShapeMatchDevApplicability_NONE, ConvDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDepthwiseFwd2D{});
};

// Tune + build + numeric-validate each distinct shape. A shape-mismatched selection regresses here.
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvCkDepthwiseFwdShapeMatch_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

// Guard against applicability expansion: a shape with no matching compiled instance is
// inapplicable.
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvCkDepthwiseFwdShapeMatchDevApplicability_NONE,
    testing::Combine(testing::Values(GetNegativeApplicabilityParams()),
                     testing::Values(GetUnsupportedShapeTestCase(miopenHalf)[0])));

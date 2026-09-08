// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the TransposedConvHipConv solver: the NCHW-capable transposing wrapper
// around the NHWC-only hipconv-backed ConvHipConv solver. It transposes the NCHW/NCDHW
// tensors to NHWC/NDHWC, runs ConvHipConv, and transposes the results back. This mirrors
// unit_conv_solver_ConvHipConv.cpp (the NHWC test) with the layout switched to NCHW.
//
// Restrictions inherited from the inner ConvHipConv::IsApplicable:
//   - 2D convolution only
//   - fp16 or bf16
//   - architectures recognised by hipconv (gfx950)
//   - the hipconv library must have a valid kernel for the (params, direction) tuple

#include "unit_conv_solver.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

namespace {

using TestCase = miopen::unit_tests::ConvTestCase;

// Small representative cases (one per channels-per-group family) for smoke runs.
// Identical logical shapes to the NHWC ConvHipConv test; only the layout tag differs so
// the transposing wrapper's NCHW->NHWC path is exercised end-to-end vs the naive reference.
auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    constexpr auto layout = miopenTensorNCHW;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64,  4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c
        TestCase{{datatype, layout, {4, 16, 8, 1}}, {datatype, layout, {16,  8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 8c
        TestCase{{datatype, layout, {4, 32, 8, 1}}, {datatype, layout, {32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 16c
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 32c
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx950);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverTransposedConvHipConvFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverTransposedConvHipConvBwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverTransposedConvHipConvWrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;

using GPU_UnitTestConvSolverTransposedConvHipConvFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverTransposedConvHipConvBwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverTransposedConvHipConvWrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvFwd_FP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvBwd_FP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvWrw_FP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvFwd_BFP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvBwd_BFP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverTransposedConvHipConvWrw_BFP16, TransposedConvHipConv)
{
    this->RunTest(miopen::solver::conv::TransposedConvHipConv{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedConvHipConvFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedConvHipConvBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedConvHipConvWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverTransposedConvHipConvFwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverTransposedConvHipConvBwd_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverTransposedConvHipConvWrw_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

#endif // MIOPEN_USE_HIPCONV

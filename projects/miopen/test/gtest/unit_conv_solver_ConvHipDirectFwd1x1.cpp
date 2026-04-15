// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT


#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // 1x1 weights, no padding, no dilation, group=1
        TestCase{{1,  16, 14, 14}, {32,  16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{4,  32, 28, 28}, {64,  32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{1,  64, 56, 56}, {64,  64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, datatype},
        TestCase{{8, 256, 14, 14}, {64, 256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.UseCpuRef();
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP32  = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverHipDirectFwd1x1DevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP16, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP16, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_BFP16, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_BFP16, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP32, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP32, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

TEST_P(CPU_UnitTestConvSolverHipDirectFwd1x1DevApplicability_NONE, ConvHipDirectFwd1x1)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd1x1{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Fwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd1x1Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipDirectFwd1x1DevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

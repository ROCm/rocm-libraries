// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Single-pass path: OUT_HEIGHT=9 which is not split across two passes.
        TestCase{{1,  1,   44,  44},  {1,  1,  11, 11}, {0, 0}, {4, 4}, {1, 1}, datatype},
        // Second-pass path: OUT_HEIGHT=55 triggers remainder row handling (last_out_extent1=3).
        TestCase{{3, 64,  227, 227}, {96, 64,  11, 11}, {0, 0}, {4, 4}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X |
                             Gpu::gfx950 | Gpu::gfx103X | Gpu::gfx110X | Gpu::gfx115X |
                             Gpu::gfx120X;
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.SetTolerance(supported_gpus, miopenFloat, 3.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipDirectFwd11x11_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwd11x11_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipDirectFwd11x11_FP32  = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverHipDirectFwd11x11DevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipDirectFwd11x11_FP16, ConvHipDirectFwd11x11)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd11x11{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd11x11_BFP16, ConvHipDirectFwd11x11)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd11x11{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwd11x11_FP32, ConvHipDirectFwd11x11)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd11x11{});
};

TEST_P(CPU_UnitTestConvSolverHipDirectFwd11x11DevApplicability_NONE, ConvHipDirectFwd11x11)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd11x11{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd11x11_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd11x11_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwd11x11_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipDirectFwd11x11DevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

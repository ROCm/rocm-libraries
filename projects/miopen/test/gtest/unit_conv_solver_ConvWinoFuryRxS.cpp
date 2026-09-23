/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1,  16, 16, 16}, { 16,  16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // c16 kernel
        TestCase{{1, 128, 16, 16}, {128, 128, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype}, // c32 kernel
        // clang-format on
    };
}

auto GetConvTestCasesWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{ 1,  16, 5, 5}, {16,  16, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype}, // c16 kernel
        TestCase{{64, 128, 7, 7}, {64, 128, 7, 7}, {0, 0}, {1, 1}, {1, 1}, datatype}, // c32 kernel
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx110X | Gpu::gfx115X |
                                                              Gpu::gfx120X);
        p.ExcludeDevice("gfx1103"); // WORKAROUND_ISSUE_3044
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverWinoFury2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverWinoFury2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverWinoFury2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using CPU_UnitTestConvSolverWinoFury2x3DevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverWinoFury2x3Fwd_FP16, ConvWinoFuryRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoFuryRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoFury2x3Bwd_FP16, ConvWinoFuryRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoFuryRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoFury2x3Wrw_FP16, ConvWinoFuryRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoFuryRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverWinoFury2x3DevApplicabilityFwd_NONE, ConvWinoFuryRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoFuryRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoFury2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoFury2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoFury2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw(miopenHalf))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverWinoFury2x3DevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

// =====================================================================
// TransposedConvWinoFuryRxS (NHWC layout)
// =====================================================================

namespace {

auto GetConvTestCasesNHWC(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{
            {datatype, miopenTensorNHWC, {1, 20, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}}
        },
        // clang-format on
    };
}

auto GetConvTestCasesNHWCWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{
            {datatype, miopenTensorNHWC, {1, 16, 5, 5}},
            {datatype, miopenTensorNHWC, {16, 16, 3, 3}},
            datatype,
            {{0, 0}, {1, 1}, {1, 1}}
        },
        // Degenerate spatial dims (H=1, W=1) with 1x1 filter that has ambiguous layout strides
        // so HeuristicUpdateLayouts() can't fix the layout string after NHWC->NCHW transposition.
        // Targets GetSwappedNCLayout(NHWC)->CHWN
        // then hits missing return in GetGroupConvLayout.
        TestCase{
            {datatype, miopenTensorNHWC, {2, 40, 1, 1}},
            {datatype, miopenTensorNHWC, {8, 40, 1, 1}},
            datatype,
            {{0, 0}, {1, 1}, {1, 1}}
        },
        // clang-format on
    };
}

} // namespace

using GPU_UnitTestConvSolverTransposedWinoFuryRxSFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverTransposedWinoFuryRxSBwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverTransposedWinoFuryRxSWrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using CPU_UnitTestConvSolverTransposedWinoFuryRxSDevApplicabilityFwd_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverTransposedWinoFuryRxSFwd_FP16, TransposedConvWinoFuryRxS)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoFuryRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedWinoFuryRxSBwd_FP16, TransposedConvWinoFuryRxS)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoFuryRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedWinoFuryRxSWrw_FP16, TransposedConvWinoFuryRxS)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoFuryRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverTransposedWinoFuryRxSDevApplicabilityFwd_FP16,
       TransposedConvWinoFuryRxS)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoFuryRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoFuryRxSFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoFuryRxSBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoFuryRxSWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWCWrw(miopenHalf))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverTransposedWinoFuryRxSDevApplicabilityFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCasesNHWC(miopenHalf)[0])));

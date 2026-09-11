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
        // rage v4.6
        TestCase{{1, 16, 135, 240}, {16, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{2,  4,  64,  64}, {16,  4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // rage v4.9
        TestCase{{1, 16, 135, 240}, {16, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{2,  4,  64,  64}, {16,  4, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        // rage v4.9 asymmetric R!=S
        TestCase{{1, 16, 135, 240}, {16, 16, 3, 5}, {1, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 16, 135, 240}, {16, 16, 5, 3}, {2, 1}, {1, 1}, {1, 1}, datatype},
        // larger C/spatial to exercise the multi-dispatch (n_groups = CUs * n_dispatches) path
        TestCase{{2, 64,  64,  64}, {64, 64, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // group convs
        TestCase{{2, 15,  28,  28}, {15,  3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 5, datatype},
        TestCase{{2, 15,  28,  28}, {15,  3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 5, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1, 16,  5,  5}, {16, 16, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 32,  7,  7}, { 4, 32, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // asymmetric R!=S
        TestCase{{1, 16,  7,  7}, {16, 16, 3, 5}, {1, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 16,  7,  7}, {16, 16, 5, 3}, {2, 1}, {1, 1}, {1, 1}, datatype},
        // group convs
        TestCase{{1, 16,  5,  5}, {16,  1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 16, datatype},
        TestCase{{2, 16, 28, 28}, {16,  1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 16, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p =
            miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx120X);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw(miopenHalf))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

// =====================================================================
// TransposedConvWinoRageRxS (NHWC layout)
// =====================================================================

namespace {

auto GetConvTestCasesNHWC(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{
            {datatype, miopenTensorNHWC, {1, 40, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 2}
        },
        // g=1 regression test for GetGroupConvLayout NHWC WrW fix
        TestCase{
            {datatype, miopenTensorNHWC, {1, 20, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}}
        },
        // asymmetric R!=S
        TestCase{
            {datatype, miopenTensorNHWC, {1, 40, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 5}},
            datatype,
            {{1, 2}, {1, 1}, {1, 1}, 2}
        },
        // clang-format on
    };
}

// WrW-specific NHWC test cases including degenerate spatial dimensions (H=1, W=1).
// With degenerate dims, NCHW strides satisfy NHWC ordering, so HeuristicUpdateLayouts()
// may fail to update the layout string after NHWC->NCHW transposition.
// This exercises the GetGroupConvLayout fix for CHWN/HWCN/HWNC layouts.
auto GetConvTestCasesNHWCWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{
            {datatype, miopenTensorNHWC, {1, 40, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 2}
        },
        // g=1 regression test for GetGroupConvLayout NHWC WrW fix
        TestCase{
            {datatype, miopenTensorNHWC, {1, 20, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}}
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
        // asymmetric R!=S
        TestCase{
            {datatype, miopenTensorNHWC, {1, 40, 20, 20}},
            {datatype, miopenTensorNHWC, {20, 20, 3, 5}},
            datatype,
            {{1, 2}, {1, 1}, {1, 1}, 2}
        },
        // clang-format on
    };
}

} // namespace

using GPU_UnitTestConvSolverTransposedWinoRageRxSFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverTransposedWinoRageRxSBwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverTransposedWinoRageRxSWrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using CPU_UnitTestConvSolverTransposedWinoRageRxSDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverTransposedWinoRageRxSFwd_FP16, TransposedConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedWinoRageRxSBwd_FP16, TransposedConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedWinoRageRxSWrw_FP16, TransposedConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoRageRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverTransposedWinoRageRxSDevApplicabilityFwd_NONE,
       TransposedConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvWinoRageRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoRageRxSFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoRageRxSBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedWinoRageRxSWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWCWrw(miopenHalf))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverTransposedWinoRageRxSDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCasesNHWC(miopenHalf)[0])));

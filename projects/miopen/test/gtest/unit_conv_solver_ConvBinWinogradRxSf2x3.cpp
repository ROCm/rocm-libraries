/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
        TestCase{
            {datatype, {1, 40, 20, 20}},
            {datatype, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 2}
        },
        // clang-format on
    };
}

// g=1 test cases are WrW-only because WORKAROUND_ISSUE_1681 rejects g=1 for Fwd/Bwd
// in ConvBinWinoRxS<2,3>. The g=1 Fwd/Bwd case is handled by ConvBinWinogradRxSf2x3g1.
auto GetConvTestCasesWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{
            {datatype, {1, 40, 20, 20}},
            {datatype, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 2}
        },
        TestCase{
            {datatype, {1, 20, 20, 20}},
            {datatype, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 1}
        },
        // clang-format on
    };
}

template <miopenDataType_t datatype>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
    Gpu supported_gpus = Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 |
                         Gpu::gfx103X | Gpu::gfx110X | Gpu::gfx115X | Gpu::gfx120X;
    if constexpr(datatype == miopenFloat)
    {
        supported_gpus = supported_gpus | Gpu::gfx900;
    }
    auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
    p.Tunable(5);
    p.CheckXnackDisabled();
    p.SetConvAttrFp16Alt(0);
    return p;
}

miopen::unit_tests::UnitTestConvSolverParams GetTestParamsHalf()
{
    return GetTestParams<miopenHalf>();
}

miopen::unit_tests::UnitTestConvSolverParams GetTestParamsFloat()
{
    return GetTestParams<miopenFloat>();
}

} // namespace

using GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32 = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// =====================================================================
// TransposedConvBinWinoRxS<2, 3> (NHWC layout)
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
        // clang-format on
    };
}

// g=1 NHWC test cases are WrW-only (WORKAROUND_ISSUE_1681 rejects g=1 Fwd/Bwd)
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
        TestCase{
            {datatype, miopenTensorNHWC, {1, 40, 20, 40}},
            {datatype, miopenTensorNHWC, {20, 40, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 1}
        },
        // Degenerate spatial dims (H=1, W=1) with 1x1 filter that has ambiguous layout strides
        // so HeuristicUpdateLayouts() can't fix the layout string after NHWC->NCHW transposition.
        // Targets GetSwappedNCLayout(NHWC)->CHWN
        // then hits missing return in GetGroupConvLayout.
        TestCase{
            {datatype, miopenTensorNHWC, {2, 40, 1, 1}},
            {datatype, miopenTensorNHWC, {8, 40, 1, 1}},
            datatype,
            {{0, 0}, {1, 1}, {1, 1}, 1}
        },
        // clang-format on
    };
}

} // namespace

using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;
using GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP32 = GPU_UnitTestConvSolverWrw_FP32;
// Note: DevApplicability tests are not included for TransposedConvBinWinoRxS<2,3> because
// the inner solver has group_count==1 restriction (WORKAROUND_ISSUE_1681) for Fwd/Bwd directions,
// which doesn't apply uniformly across all devices listed in supported_gpus

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP16, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP16, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP16, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP32, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP32, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP32, TransposedConvBinWinoRxSf2x3)
{
    this->RunTest(miopen::solver::conv::TransposedConvBinWinoRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWCWrw(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Fwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWC(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverTransposedBinWinogradRxSf2x3Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesNHWCWrw(miopenFloat))));

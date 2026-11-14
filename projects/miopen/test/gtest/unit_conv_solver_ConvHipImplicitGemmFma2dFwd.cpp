/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

auto GetConvTestCases(miopenDataType_t datatype, miopenTensorLayout_t layout)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // x {N, C, H, W}, w {K, C / g, Y, X}, pad / stride / dilation {H, W}.
    return std::vector{
        // clang-format off
        TestCase{{1, 32, 16, 16}, {32, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype, layout},
        // Odd channel counts, partial M / N tiles, per-dimension stride and padding.
        TestCase{{2, 17, 10, 7}, {9, 17, 1, 2}, {0, 1}, {1, 2}, {1, 1}, datatype, layout},
        TestCase{{1, 64, 8, 8}, {130, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype, layout},
        // Dilated.
        TestCase{{1, 24, 9, 11}, {40, 24, 3, 3}, {2, 3}, {1, 1}, {2, 3}, datatype, layout},
        // Grouped, 16 output channels per group (the solver's lower bound).
        TestCase{{1, 32, 8, 8}, {64, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, datatype, layout},
        // Grouped and dilated, partial channel chunk (20 input channels per group).
        TestCase{{2, 40, 13, 12}, {32, 20, 3, 3}, {2, 2}, {1, 1}, {2, 2}, 2, datatype, layout},
        TestCase{{1, 64, 10, 10}, {64, 16, 3, 3}, {4, 4}, {2, 1}, {4, 4}, 4, datatype, layout},
        // clang-format on
    };
}

auto GetAllConvTestCases(miopenDataType_t datatype)
{
    auto cases = GetConvTestCases(datatype, miopenTensorNCHW);
    for(auto&& c : GetConvTestCases(datatype, miopenTensorNHWC))
        cases.push_back(std::move(c));
    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(
            Gpu::gfx103X | Gpu::gfx110X | Gpu::gfx115X | Gpu::gfx117X | Gpu::gfx120X);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;

using CPU_UnitTestConvSolverHipImplicitGemmFma2dFwdDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP32, ConvHipImplicitGemmFma2dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFma2dFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP16, ConvHipImplicitGemmFma2dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFma2dFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_BFP16, ConvHipImplicitGemmFma2dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFma2dFwd{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmFma2dFwdDevApplicabilityFwd_NONE,
       ConvHipImplicitGemmFma2dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmFma2dFwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetAllConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetAllConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmFma2dFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetAllConvTestCases(miopenBFloat16))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemmFma2dFwdDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat,
                                                                           miopenTensorNCHW)[0])));

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

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // 1D convolutions as NCHW with H == 1: x {N, C, 1, W}, w {K, C / g, 1, R}.
    return std::vector{
        // clang-format off
        TestCase{{1, 128, 1, 4096}, {128, 128, 1, 3}, {0, 3}, {1, 1}, {1, 3}, datatype},
        TestCase{{2, 96, 1, 1000}, {64, 96, 1, 7}, {0, 27}, {1, 2}, {1, 9}, datatype},
        TestCase{{3, 17, 1, 777}, {33, 17, 1, 3}, {0, 2}, {1, 3}, {1, 2}, datatype},
        TestCase{{1, 64, 1, 4099}, {64, 16, 1, 5}, {0, 2}, {1, 1}, {1, 1}, 4, datatype},
        TestCase{{1, 32, 1, 2048}, {32, 32, 1, 11}, {0, 0}, {1, 1}, {1, 5}, datatype},
        // Ungrouped with few channels (BigVGAN-style last stage).
        TestCase{{1, 8, 1, 3000}, {8, 8, 1, 11}, {0, 25}, {1, 1}, {1, 5}, datatype},
        // Near the LDS limit: (8 * (1013 + 16 * 64)) floats = 65184 bytes.
        TestCase{{1, 16, 1, 2048}, {16, 16, 1, 16}, {0, 0}, {1, 1}, {1, 59}, datatype},
        // clang-format on
    };
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

using GPU_UnitTestConvSolverHipDirectConv1dFwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverHipDirectConv1dFwdDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipDirectConv1dFwd_FP32, ConvHipDirectConv1dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectConv1dFwd{});
};

TEST_P(CPU_UnitTestConvSolverHipDirectConv1dFwdDevApplicabilityFwd_NONE, ConvHipDirectConv1dFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectConv1dFwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectConv1dFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipDirectConv1dFwdDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

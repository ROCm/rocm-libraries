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
        TestCase{{1, 16, 14, 14}, {16, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        // gfx10 and gfx11 are disabled due to WORKAROUND_SWDEV_266868
        Gpu supported_gpus = Gpu::All & ~(Gpu::gfx103X | Gpu::gfx110X);
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverOclBwdWrW1x1_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverOclBwdWrW1x1_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverOclBwdWrW1x1_FP32  = GPU_UnitTestConvSolverWrw_FP32;

using CPU_UnitTestConvSolverOclBwdWrW1x1DevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverOclBwdWrW1x1_FP16, ConvOclBwdWrW1x1)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW1x1{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW1x1_BFP16, ConvOclBwdWrW1x1)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW1x1{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW1x1_FP32, ConvOclBwdWrW1x1)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW1x1{});
};

TEST_P(CPU_UnitTestConvSolverOclBwdWrW1x1DevApplicability_NONE, ConvOclBwdWrW1x1)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW1x1{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW1x1_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW1x1_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW1x1_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverOclBwdWrW1x1DevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

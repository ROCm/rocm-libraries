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
#include "get_handle.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // dilation=2: effective kernel size 5, pad=2 keeps spatial dims at 8
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {2, 2}, {1, 1}, {2, 2}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    if(datatype == miopenHalf)
    {
        const miopen::Handle& handle = get_handle();
        const std::string name       = handle.GetDeviceName();

        // clang-format off
        if(!miopen::StartsWith(name, "gfx1151"))
        {
            // Regression test for https://github.com/ROCm/MIOpen/issues/1956
            cases.emplace_back(TestCase{{2, 64, 128, 128, 128}, {32, 64, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenHalf});
        }
        else
        {
            //https://github.com/ROCm/TheRock/issues/3202
            // We have out of memory error on gfx1151 (flaky issue), so let's reduce tensor size for this type of machibes
            cases.emplace_back(TestCase{{2, 64, 64, 64, 64}, {32, 64, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenHalf});
        }
        // clang-format on
    }

    // clang-format off
    // dilation=3: effective kernel size 7, pad=3 keeps spatial dims at 8
    cases.emplace_back(TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {3, 3}, {1, 1}, {3, 3}, datatype});
    // asymmetric dilation: different dilation per spatial dim
    cases.emplace_back(TestCase{{1, 8, 9, 9}, {8, 8, 3, 3}, {1, 2}, {1, 1}, {1, 2}, datatype});
    // 3D dilation=2: effective kernel size 5 in each dim, pad=2 keeps spatial dims at 8
    cases.emplace_back(TestCase{{1, 8, 8, 8, 8}, {8, 8, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, datatype});
    // 3D asymmetric dilation: dilation=(1,2,3), pad=(1,2,3) keeps spatial dims at 8
    cases.emplace_back(TestCase{{1, 8, 8, 8, 8}, {8, 8, 3, 3, 3}, {1, 2, 3}, {1, 1, 1}, {1, 2, 3}, datatype});
    // clang-format on

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverGemmBwdRestBwd_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverGemmBwdRestBwd_FP32  = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_FP16, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmBwdRestBwd_FP32, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDevApplicabilityGemmBwdRestBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmBwdRestBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));

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

    auto cases = std::vector<TestCase>{};

    for(int in_hw : {7, 14, 27, 28})
    {
        // clang-format off
        cases.emplace_back(TestCase{{1, 16, in_hw, in_hw}, {48, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{4, 16, in_hw, in_hw}, {16, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{1, 32, in_hw, in_hw}, {16, 32, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{1, 16, in_hw, in_hw}, {64, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{64, 16, in_hw, in_hw}, {16, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{64, 16, in_hw, in_hw}, {64, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{64, 1, in_hw, in_hw}, {16, 1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype});
        // clang-format on
    }

    return cases;
}

// Half-open [begin, end) slice of the conv cases. The Smoke/Standard/Full tiers
// use disjoint slices; the categories run cumulative tiers (standard =
// Smoke+Standard, comprehensive/full = Smoke+Standard+Full), so the slices
// reassemble the full list with no case repeated across tiers.
auto GetConvTestCasesRange(miopenDataType_t datatype, std::size_t begin, std::size_t end)
{
    auto cases = GetConvTestCases(datatype);
    if(begin > cases.size())
        begin = cases.size();
    if(end > cases.size())
        end = cases.size();
    if(begin > end)
        begin = end;
    return decltype(cases)(cases.begin() + begin, cases.begin() + end);
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverFFTFwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverFFTBwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverFFTDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverFFTFwd_FP32, fft) { this->RunTest(miopen::solver::conv::fft{}); };

TEST_P(GPU_UnitTestConvSolverFFTBwd_FP32, fft) { this->RunTest(miopen::solver::conv::fft{}); };

TEST_P(CPU_UnitTestConvSolverFFTDevApplicabilityFwd_NONE, fft)
{
    this->RunTest(miopen::solver::conv::fft{});
};

// Tiered with disjoint conv-case slices: Smoke [0,3), Standard [3,10), Full
// [10,end). The categories run cumulative tiers, so Smoke+Standard+Full
// reassemble the complete list with no case repeated across tiers.
constexpr std::size_t kAllConvCases = static_cast<std::size_t>(-1);
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverFFTFwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 0, 3))));
INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverFFTFwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 3, 10))));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverFFTFwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 10, kAllConvCases))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverFFTBwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 0, 3))));
INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverFFTBwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 3, 10))));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverFFTBwd_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoFFT),
                     testing::ValuesIn(GetConvTestCasesRange(miopenFloat, 10, kAllConvCases))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverFFTDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

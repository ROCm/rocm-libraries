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

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    auto cases = std::vector{
        // clang-format off
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y},
        TestCase{{2, 8, 11, 9}, {12, 4, 3, 3}, {1, 1}, {2, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC, miopenTensorNHWC, 2},
        TestCase{{1, 4, 7, 8, 9}, {6, 4, 3, 2, 3}, {1, 0, 1}, {1, 2, 1}, {1, 1, 2}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC},
        // clang-format on
    };

    // Point-output shapes (stride == filter, output spatially 1x1) take the single-GEMM path.
    // FP32 is left out: at K=1280 its RMS error sits just under the 1.0*eps threshold that
    // non-TF32 GPUs use, so the case is not reliable there.
    if(datatype == miopenHalf)
    {
        // clang-format off
        cases.emplace_back(TestCase{{4, 3, 14, 14}, {1280, 3, 14, 14}, {0, 0}, {14, 14}, {1, 1}, type_x, type_w, type_y});
        cases.emplace_back(TestCase{{4, 3, 4, 4, 4}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, type_x, type_w, type_y});
        cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {4, 4, 14, 14}},
                                    {datatype, miopenTensorNHWC, {64, 4, 14, 14}},
                                    datatype, {{0, 0}, {14, 14}, {1, 1}}});
        cases.emplace_back(TestCase{{datatype, miopenTensorNDHWC, {4, 4, 4, 4, 4}},
                                    {datatype, miopenTensorNDHWC, {64, 4, 4, 4, 4}},
                                    datatype, {{0, 0, 0}, {4, 4, 4}, {1, 1, 1}}});
        // clang-format on
    }

    return cases;
}

// Point-output bf16, in 2D and 3D, exercising the single-GEMM path with a bf16 GEMM.
auto GetConvTestCasesPointOutputBf16()
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    constexpr auto datatype = miopenBFloat16;

    return std::vector{
        // clang-format off
        TestCase{{4, 3, 14, 14}, {1280, 3, 14, 14}, {0, 0}, {14, 14}, {1, 1}, datatype, datatype, datatype},
        TestCase{{4, 3, 4, 4, 4}, {512, 3, 4, 4, 4}, {0, 0, 0}, {4, 4, 4}, {1, 1, 1}, datatype, datatype, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    return std::vector{
        // clang-format off
        // Regression test for https://github.com/ROCm/MIOpen/issues/2047
        TestCase{{1, 1, 2, 1, 2}, {2, 1, 2, 1, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y},
        // clang-format on
    };
}

auto GetConvTestCasesIntMaxOverflow()
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // Per-group im2col extent: 15447 * 15447 * 3 * 3 = 2,147,488,281 > INT_MAX.
    return std::vector{TestCase{{1, 2, 15449, 15449},
                                {2, 1, 3, 3},
                                {0, 0},
                                {1, 1},
                                {1, 1},
                                miopenInt8,
                                miopenInt8,
                                miopenInt32,
                                miopenTensorNHWC,
                                miopenTensorNHWC,
                                2}};
}

const auto& GetOverflowTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.UseGpuRef();
        return p;
    }();
    return params;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        return p;
    }();
    return params;
}

// BFP16 Full tests include 3D convolution shapes that trigger a rocBLAS bug on gfx90a
// (rocBLAS does not support BF16->BF16 GEMM on that architecture). Skip on gfx90A only;
// all other GPUs continue to exercise the full test suite.
// TODO: Remove this exclusion once the rocBLAS bug is fixed.
const auto& GetTestParamsNoGfx90A()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All & ~Gpu::gfx90A);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverGemmFwdRestFwd_FP16             = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16            = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverGemmFwdRestFwd_FP32             = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverGemmFwdRestFwd_I8               = GPU_UnitTestConvSolverFwd_I8;
using GPU_UnitTestConvSolverGemmFwdRestIntMaxOverflowFwd_I8 = GPU_UnitTestConvSolverFwd_I8;
using CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_FP16, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_FP32, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_I8, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestIntMaxOverflowFwd_I8, GemmFwdRest)
{
    constexpr std::size_t minimum_device_memory = 16ULL << 30;
    if(get_handle().GetGlobalMemorySize() < minimum_device_memory)
        GTEST_SKIP() << "Requires at least 16 GiB of device memory";

    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST(CPU_UnitTestConvSolverGemmFwdRestFwd_NONE, RejectsUnsupportedInt8Output)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    const auto test_case = TestCase{
        {1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, miopenInt8, miopenInt8, miopenHalf};
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::Forward);
    auto context       = miopen::ExecutionContext{&get_handle()};
    problem.SetupFloats(context);
    problem.SetupComputeType(context);

    EXPECT_FALSE(miopen::solver::conv::GemmFwdRest{}.IsApplicable(context, problem));
}

TEST(CPU_UnitTestConvSolverGemmFwdRestFwd_NONE, RejectsNonInt8WeightsForInt8Input)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    const auto test_case = TestCase{
        {1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, miopenInt8, miopenFloat, miopenFloat};
    const auto y_desc  = miopen::TensorDescriptor(miopenFloat, {1, 8, 6, 6});
    const auto problem = miopen::conv::ProblemDescription(test_case.GetXTensorDescriptor(),
                                                          test_case.GetWTensorDescriptor(),
                                                          y_desc,
                                                          test_case.GetConv(),
                                                          miopen::conv::Direction::Forward);
    auto context       = miopen::ExecutionContext{&get_handle()};

    EXPECT_FALSE(miopen::solver::conv::GemmFwdRest{}.IsApplicable(context, problem));
}

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokePointOutput,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParamsNoGfx90A()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesPointOutputBf16())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenInt8))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParamsNoGfx90A()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenInt8))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestIntMaxOverflowFwd_I8,
                         testing::Combine(testing::Values(GetOverflowTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesIntMaxOverflow())));

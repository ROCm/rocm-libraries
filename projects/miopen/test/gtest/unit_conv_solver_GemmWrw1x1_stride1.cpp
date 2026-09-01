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
        TestCase{{1, 8, 8, 8}, {8, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

// Channel-last pointwise shapes, which take the single unbatched GEMM path.
auto GetConvTestCasesNhwc(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {4, 16, 14, 14}},
                 {datatype, miopenTensorNHWC, {32, 16, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        TestCase{{datatype, miopenTensorNDHWC, {4, 16, 4, 4, 4}},
                 {datatype, miopenTensorNDHWC, {32, 16, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}},
        // A 1x1 input is also point-output, so GemmWrwUniversal claims it as well.
        TestCase{{datatype, miopenTensorNHWC, {4, 16, 1, 1}},
                 {datatype, miopenTensorNHWC, {32, 16, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        // clang-format on
    };
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

const auto& GetTestParamsNhwc()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 2.0f);
        p.SetTolerance(Gpu::gfx90A, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

// rocBLAS does not support BF16->BF16 GEMM on gfx90a, so skip bf16 there.
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

using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32  = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

TEST_P(CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE, GemmWrw1x1_stride1)
{
    this->RunTest(miopen::solver::conv::GemmWrw1x1_stride1{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Channel-last smoke tests
INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_BFP16,
                         testing::Combine(testing::Values(GetTestParamsNoGfx90A()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         GPU_UnitTestConvSolverGemmWrw1x1Stride1Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesNhwc(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

INSTANTIATE_TEST_SUITE_P(SmokeNhwc,
                         CPU_UnitTestConvSolverGemmWrw1x1Stride1DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParamsNhwc()),
                                          testing::Values(GetConvTestCasesNhwc(miopenFloat)[0])));

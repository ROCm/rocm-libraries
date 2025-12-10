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

// These tests verify that GEMM solvers work correctly with large memory allocations
// after the removal of the 7.2GB VRAM limit.
// All test configurations require workspace memory exceeding 8 GB for FP32.

namespace {

// Test cases for GemmFwdRest (non-1x1 kernels or stride 1)
auto GetGemmFwdRestTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    return std::vector{
        // clang-format off
        // Config: N=96, C=256, H=112, W=112, K=256, R=3, S=3, stride=1, pad=1
        // Input: 96*256*112*112 = 308,281,344 elements
        // Output: 96*256*112*112 = 308,281,344 elements
        // Im2col workspace: 96*256*9*112*112 = 2,774,532,096 elements
        // Workspace: ~11.1 GB for FP32, ~5.5 GB for FP16 (exceeds 8 GB requirement for FP32, under 12 GB)
        TestCase{{96, 256, 112, 112}, {256, 256, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

// Test cases for GemmBwdRest (non-1x1 kernels)
auto GetGemmBwdRestTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    return std::vector{
        // clang-format off
        // Config: N=96, C=256, H=112, W=112, K=256, R=3, S=3, stride=1, pad=1
        // dy (output grad): 96*256*112*112 = 308,281,344 elements
        // dx (input grad): 96*256*112*112 = 308,281,344 elements
        // Col2im workspace: 96*256*9*112*112 = 2,774,532,096 elements
        // Workspace: ~11.1 GB for FP32, ~5.5 GB for FP16 (exceeds 8 GB requirement for FP32, under 12 GB)
        TestCase{{96, 256, 112, 112}, {256, 256, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

// Test cases for GemmWrwUniversal
auto GetGemmWrwUniversalTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;
    return std::vector{
        // clang-format off
        // Config: N=96, C=256, H=112, W=112, K=256, R=3, S=3, stride=1, pad=1
        // x (input): 96*256*112*112 = 308,281,344 elements
        // dy (output grad): 96*256*112*112 = 308,281,344 elements
        // dw (weight grad): 256*256*3*3 = 589,824 elements
        // Workspace: 96*256*9*112*112 = 2,774,532,096 elements
        // Workspace: ~11.1 GB for FP32, ~5.5 GB for FP16 (exceeds 8 GB requirement for FP32, under 12 GB)
        TestCase{{96, 256, 112, 112}, {256, 256, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        // Verify on GPU
        p.UseGpuRef();
        // Increase tolerance for FP32 due to large tensor accumulation errors
        p.SetTolerance(Gpu::All, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

//************************************************************************************
// GemmFwdRest Tests
//************************************************************************************
using GPU_GemmFwdRest_LargeMemory_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_GemmFwdRest_LargeMemory_FP16 = GPU_UnitTestConvSolverFwd_FP16;

TEST_P(GPU_GemmFwdRest_LargeMemory_FP32, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_GemmFwdRest_LargeMemory_FP16, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GemmFwdRest_LargeMemory_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetGemmFwdRestTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GemmFwdRest_LargeMemory_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetGemmFwdRestTestCases(miopenHalf))));

//************************************************************************************
// GemmBwdRest Tests
//************************************************************************************
using GPU_GemmBwdRest_LargeMemory_FP32 = GPU_UnitTestConvSolverBwd_FP32;
using GPU_GemmBwdRest_LargeMemory_FP16 = GPU_UnitTestConvSolverBwd_FP16;

TEST_P(GPU_GemmBwdRest_LargeMemory_FP32, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

TEST_P(GPU_GemmBwdRest_LargeMemory_FP16, GemmBwdRest)
{
    this->RunTest(miopen::solver::conv::GemmBwdRest{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GemmBwdRest_LargeMemory_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetGemmBwdRestTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GemmBwdRest_LargeMemory_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetGemmBwdRestTestCases(miopenHalf))));

//************************************************************************************
// GemmWrwUniversal Tests
//************************************************************************************
using GPU_GemmWrwUniversal_LargeMemory_FP32 = GPU_UnitTestConvSolverWrw_FP32;
using GPU_GemmWrwUniversal_LargeMemory_FP16 = GPU_UnitTestConvSolverWrw_FP16;

TEST_P(GPU_GemmWrwUniversal_LargeMemory_FP32, GemmWrwUniversal)
{
    this->RunTest(miopen::solver::conv::GemmWrwUniversal{});
};

TEST_P(GPU_GemmWrwUniversal_LargeMemory_FP16, GemmWrwUniversal)
{
    this->RunTest(miopen::solver::conv::GemmWrwUniversal{});
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_GemmWrwUniversal_LargeMemory_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoGEMM),
                     testing::ValuesIn(GetGemmWrwUniversalTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_GemmWrwUniversal_LargeMemory_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoGEMM),
                     testing::ValuesIn(GetGemmWrwUniversalTestCases(miopenHalf))));

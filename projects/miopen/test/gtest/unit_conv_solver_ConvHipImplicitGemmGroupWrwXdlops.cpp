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

#define WORKAROUND_SWDEV_522871 1

#if WORKAROUND_SWDEV_522871
#define SOLVER_NAME_DEV_APP DISABLED_ConvHipImplicitGemmGroupWrwXdlops
#else
#define SOLVER_NAME_DEV_APP ConvHipImplicitGemmGroupWrwXdlops
#endif

namespace {

auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off

        TestCase{{8, 8, 3, 3}, {8, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, datatype},

        // clang-format on
    };
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                 {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}}}, // non-zero padding
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                 {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                 datatype, {{0, 0}, {2, 2}, {1, 1}}}, // stride > 1

        // Group count = 2 and 4                 
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                 {datatype, miopenTensorNHWC, {96, 32, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {2, 2}, 2}}, // dilation > 1

        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                 {datatype, miopenTensorNHWC, {96, 16, 1, 1}},
                 datatype, {{0, 0}, {2, 2}, {1, 1}, 4}}, // stride > 1

        // clang-format on
    };
}

const auto& GetTestParams(bool bfp16 = false)
{
    static const auto params = [bfp16] {
        Gpu supportedDevices = Gpu::None;
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        if(!bfp16)
            supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
        else
            supportedDevices = Gpu::gfx94X | Gpu::gfx950;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_I8    = GPU_UnitTestConvSolverWrw_I8;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32  = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_I8, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_NONE, SOLVER_NAME_DEV_APP)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenInt8))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams(true)),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat))));

// Full tests
/*INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenInt8))));*/

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams(true)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

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

#ifndef HIP_PACKAGE_VERSION_FLAT
#error "HIP_PACKAGE_VERSION_FLAT undefined"
#endif

// LLVM buffer intrinsics llvm.amdgcn.buffer.* have been removed in HIP 6.4
#define WORKAROUND_SWDEV_498660 (HIP_PACKAGE_VERSION_FLAT >= 6004000000)

#if WORKAROUND_SWDEV_498660
#define SOLVER_NAME DISABLED_ConvHipImplicitGemmWrwV4R4XdlopsPaddedGemm
#else
#define SOLVER_NAME ConvHipImplicitGemmWrwV4R4XdlopsPaddedGemm
#endif

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{256, 8, 5, 5}, {8, 8, 3, 3}, {1, 1}, {2, 2}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        p.SetConvAttrFp16Alt(0);
        p.SetTolerance(Gpu::gfx90A, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP16 =
    GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_BFP16 =
    GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP32 =
    GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmDevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_BFP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP32, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmDevApplicabilityWrw_NONE,
       SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmWrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmWrwV4R4XdlopsPaddedGemmDevApplicabilityWrw_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenFloat)[0])));

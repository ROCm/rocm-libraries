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

// MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1=1 is necessary due to
// WORKAROUND_SWDEV_229277_227616_229195, which disables ConvHipImplicitGemmBwdDataV4R1, but we
// still want to check that the solver is not broken.
#define WORKAROUND_SWDEV_229277_227616_229195 1

// LLVM buffer intrinsics llvm.amdgcn.buffer.* have been removed in HIP 6.4
#define WORKAROUND_SWDEV_498660 (HIP_PACKAGE_VERSION_FLAT >= 6004000000)

#if WORKAROUND_SWDEV_498660
#define SOLVER_NAME DISABLED_ConvHipImplicitGemmBwdDataV4R1
#else
#define SOLVER_NAME ConvHipImplicitGemmBwdDataV4R1
#endif

#if WORKAROUND_SWDEV_229277_227616_229195
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1)
#endif

namespace {

#if WORKAROUND_SWDEV_229277_227616_229195
class SolverEnabler
{
public:
    SolverEnabler()
    {
        if(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1)
            prev = lib_env::value<bool>(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1);
        if(prev != true)
            lib_env::update(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1, true);
    }

    ~SolverEnabler()
    {
        if(prev)
        {
            if(prev != true)
                lib_env::update(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1, false);
        }
        else
        {
            lib_env::clear(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1);
        }
    }

private:
    std::optional<bool> prev;
};
#endif

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 64, 16, 16}, {64, 64, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus =
            Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx103X;
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1Bwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;
using CPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1DevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1Bwd_FP32, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_229277_227616_229195
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1DevApplicabilityBwd_NONE, SOLVER_NAME)
{
#if WORKAROUND_SWDEV_229277_227616_229195
    SolverEnabler solver_enabler;
#endif
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemmBwdDataV4R1DevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

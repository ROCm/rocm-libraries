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
#include "../lib_env_var.hpp"

#include <limits>

MIOPEN_LIB_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)
MIOPEN_LIB_ENV_VAR(MIOPEN_TUNING_TIME_MS_MAX)

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 16, 16, 16}, {16, 16, 5, 5}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx103X;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.Tunable(5);
        return p;
    }();
    return params;
}

const auto& GetParallelCompileTestParams()
{
    static const auto params = [] {
        auto p = GetTestParams();
        p.Tunable(16);
        return p;
    }();
    return params;
}

const auto& GetTimeBudgetTestParams()
{
    static const auto params = [] {
        auto p = GetTestParams();
        p.Tunable(std::numeric_limits<std::size_t>::max());
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipDirectFwdFwd_FP16                = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwdBwd_FP16                = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwdParallelCompileFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwdTimeBudgetFwd_FP16      = GPU_UnitTestConvSolverFwd_FP16;

using GPU_UnitTestConvSolverHipDirectFwdFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipDirectFwdBwd_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;

using GPU_UnitTestConvSolverHipDirectFwdFwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverHipDirectFwdBwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;

using CPU_UnitTestConvSolverHipDirectFwdDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipDirectFwdFwd_FP16, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwdBwd_FP16, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

// Candidate kernels are built on several threads before they are timed.
TEST_P(GPU_UnitTestConvSolverHipDirectFwdParallelCompileFwd_FP16, ConvHipDirectFwd)
{
    ScopedEnvironment<int> compile_parallel_level(MIOPEN_COMPILE_PARALLEL_LEVEL, 4);
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

// The search stops when MIOPEN_TUNING_TIME_MS_MAX is exceeded and still returns a valid config.
TEST_P(GPU_UnitTestConvSolverHipDirectFwdTimeBudgetFwd_FP16, ConvHipDirectFwd)
{
    ScopedEnvironment<int> tuning_time_max(MIOPEN_TUNING_TIME_MS_MAX, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 4);
    testing::internal::CaptureStderr();
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("tuning time budget"), std::string::npos) << log;
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwdFwd_BFP16, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwdBwd_BFP16, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwdFwd_FP32, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

TEST_P(GPU_UnitTestConvSolverHipDirectFwdBwd_FP32, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

TEST_P(CPU_UnitTestConvSolverHipDirectFwdDevApplicabilityFwd_NONE, ConvHipDirectFwd)
{
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdParallelCompileFwd_FP16,
                         testing::Combine(testing::Values(GetParallelCompileTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdTimeBudgetFwd_FP16,
                         testing::Combine(testing::Values(GetTimeBudgetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipDirectFwdDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

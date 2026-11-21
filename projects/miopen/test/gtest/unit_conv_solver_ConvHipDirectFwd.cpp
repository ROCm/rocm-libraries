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
#include "../lib_env_var.hpp"

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>

#include <cstdint>
#include <limits>

MIOPEN_LIB_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)
MIOPEN_LIB_ENV_VAR(MIOPEN_TUNING_TIME_MS_MAX)
MIOPEN_LIB_ENV_VAR(MIOPEN_TUNING_PATIENCE)

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

const auto& GetFullSearchTestParams()
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
using GPU_UnitTestConvSolverHipDirectFwdPatienceFwd_FP16        = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwdFailedMeasureFwd_FP16   = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipDirectFwdHugeTimeBudgetFwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;

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

// The search stops after MIOPEN_TUNING_PATIENCE runs without improvement and still returns a
// valid config.
TEST_P(GPU_UnitTestConvSolverHipDirectFwdPatienceFwd_FP16, ConvHipDirectFwd)
{
    // The time budget is far away, so only the patience can end the search.
    ScopedEnvironment<int> tuning_time_max(MIOPEN_TUNING_TIME_MS_MAX, 3600000);
    ScopedEnvironment<int> tuning_patience(MIOPEN_TUNING_PATIENCE, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 4);
    testing::internal::CaptureStderr();
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("no improvement in 1 runs"), std::string::npos) << log;
    EXPECT_EQ(log.find("tuning time budget"), std::string::npos) << log;
};

// A budget too large for a nanosecond count must not end the search early.
TEST_P(GPU_UnitTestConvSolverHipDirectFwdHugeTimeBudgetFwd_FP16, ConvHipDirectFwd)
{
    ScopedEnvironment<std::uint64_t> tuning_time_max(MIOPEN_TUNING_TIME_MS_MAX, 10'000'000'000'000);
    ScopedEnvironment<int> tuning_patience(MIOPEN_TUNING_PATIENCE, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 4);
    testing::internal::CaptureStderr();
    this->RunTest(miopen::solver::conv::ConvHipDirectFwd{});
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("no improvement in 1 runs"), std::string::npos) << log;
    EXPECT_EQ(log.find("tuning time budget"), std::string::npos) << log;
};

// The search stops when MIOPEN_TUNING_TIME_MS_MAX is exceeded even if no measurement succeeds.
// The search gets no bias buffer, so it fails to measure every candidate of a problem with bias.
TEST_P(GPU_UnitTestConvSolverHipDirectFwdFailedMeasureFwd_FP16, ConvHipDirectFwd)
{
    // The first batch of kernels takes longer to build than the budget.
    ScopedEnvironment<int> compile_parallel_level(MIOPEN_COMPILE_PARALLEL_LEVEL, 4);
    ScopedEnvironment<int> tuning_time_max(MIOPEN_TUNING_TIME_MS_MAX, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 4);

    const auto& conv_config = std::get<2>(GetParam());
    const auto x_desc       = conv_config.GetXTensorDescriptor();
    const auto w_desc       = conv_config.GetWTensorDescriptor();
    const auto conv_desc    = conv_config.GetConv();
    const auto y_desc =
        conv_desc.GetForwardOutputTensor(x_desc, w_desc, conv_config.GetYDataType());
    const auto problem = miopen::conv::ProblemDescription(
        x_desc, w_desc, y_desc, conv_desc, miopen::conv::Direction::Forward, 1);

    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const miopen::solver::conv::ConvHipDirectFwd solver{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));

    // The search never launches a kernel here, so the buffers stay uninitialized.
    const auto x_dev = handle.Create(x_desc.GetNumBytes());
    const auto w_dev = handle.Create(w_desc.GetNumBytes());
    const auto y_dev = handle.Create(y_desc.GetNumBytes());
    const auto tensors =
        miopen::ConvFwdTensors{x_desc, x_dev.get(), w_desc, w_dev.get(), y_desc, y_dev.get()};
    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, nullptr, 0, conv_desc.attribute.gfx90aFp16alt.GetFwd()};

    testing::internal::CaptureStderr();
    EXPECT_THROW(solver.FindSolutionSimple(ctx, problem, invoke_params), miopen::Exception);
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("Bias buffer required"), std::string::npos) << log;
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
                         testing::Combine(testing::Values(GetFullSearchTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdPatienceFwd_FP16,
                         testing::Combine(testing::Values(GetFullSearchTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdHugeTimeBudgetFwd_FP16,
                         testing::Combine(testing::Values(GetFullSearchTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipDirectFwdFailedMeasureFwd_FP16,
                         testing::Combine(testing::Values(GetFullSearchTestParams()),
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

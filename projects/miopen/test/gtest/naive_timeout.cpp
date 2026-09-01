// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for MIOPEN_NAIVE_TIMEOUT behavior in the find path.
//
// When MIOPEN_NAIVE_TIMEOUT=1 (default) and a non-naive solver has already
// succeeded, the naive solver's warmup run is launched on a disposable stream
// with a wall-clock budget of 3x the best non-naive time.  If the warmup
// exceeds that budget the stream is abandoned and the naive solver is skipped.
// If the warmup finishes in time, naive proceeds through normal benchmarking
// and may appear in (or even win) the results.
//
// Paths through EvaluateInvokers naive-timeout logic (solver_finders.cpp):
//
//   Path  naive_timeout  defer_naive  non_naive_succeeded  Outcome
//   ----  -------------  -----------  -------------------  -------------------------
//   A     false          -            -                    naive evaluated (flag off)
//   B     true           false        -                    naive evaluated (no non-naive exists)
//   C     true           true         false                naive evaluated (non-naive all failed)
//   D     true           true         true                 naive timed — may pass or be skipped
//
// Tests cover paths A, B, and D. Path C requires fault injection.
//
// Path B uses MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd to restrict
// the candidate set to the naive solver only.
//
// Tests assert presence/absence of naive solvers rather than which solver wins
// by timing, so results are not sensitive to GPU performance variation.
//
// Each GPU test uses ScopedFindDb to disable miopen::debug::testing_find_db_enabled
// so TryLoad always calls the regenerator (FindCore / EvaluateInvokers).

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/solver_id.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../tensor_holder.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_TIMEOUT)
MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_TIMEOUT_FACTOR)

namespace {

struct ScopedFindDb
{
    ScopedFindDb() : prev_(miopen::debug::testing_find_db_enabled)
    {
        miopen::debug::testing_find_db_enabled = false;
    }
    ~ScopedFindDb() { miopen::debug::testing_find_db_enabled = prev_; }

    bool prev_;
};

std::vector<std::string> RunFind(miopenHandle_t handle, size_t max_solutions = 8)
{
    tensor<float> x{1, 64, 14, 14};
    tensor<float> w{64, 64, 3, 3};

    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    tensor<float> y{conv.GetForwardOutputTensor(x.desc, w.desc)};

    miopenProblem_t problem = nullptr;
    EXPECT_EQ(miopenCreateConvProblem(&problem, &conv, miopenProblemDirectionForward),
              miopenStatusSuccess);
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
              miopenStatusSuccess);
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionW, &w.desc),
              miopenStatusSuccess);
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionY, &y.desc),
              miopenStatusSuccess);

    std::vector<miopenSolution_t> solutions(max_solutions);
    size_t num_found = 0;
    EXPECT_EQ(
        miopenFindSolutions(handle, problem, nullptr, solutions.data(), &num_found, max_solutions),
        miopenStatusSuccess);
    solutions.resize(num_found);

    std::vector<std::string> names;
    for(auto sol : solutions)
    {
        uint64_t solver_id = 0;
        EXPECT_EQ(miopenGetSolutionSolverId(sol, &solver_id), miopenStatusSuccess);
        names.push_back(miopen::solver::Id{solver_id}.ToString());
        miopenDestroySolution(sol);
    }

    miopenDestroyProblem(problem);
    return names;
}

bool IsNaive(const std::string& name) { return name.find("Naive") != std::string::npos; }

// RunFind's convolution (3x3, pad 1, stride 1, fp32, K=64) is exactly what
// ConvBinWinograd3x3U accepts, and that solver is applicable only on these four
// archs.  There IsWinograd3x3SupportedAndFast() sets use_winograd_only, which
// disables the Direct finder outright, so ConvDirectNaive is never a candidate
// and assertions about its presence or absence say nothing about the timeout.
bool IsWinogradOnlyArch(const miopen::Handle& handle)
{
    const auto name = handle.GetDeviceName();
    return name == "gfx803" || name == "gfx900" || name == "gfx906" || name == "gfx908";
}

} // namespace

// With MIOPEN_NAIVE_TIMEOUT=0 (opt-out), a naive solver must appear somewhere in the
// results, confirming that the timeout logic did not fire.
TEST(GPU_NaiveTimeout_FP32, NaivePresentWhenTimeoutDisabled)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, false);
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty()) << "miopenFindSolutions returned no results";
    const bool any_naive = std::any_of(solvers.begin(), solvers.end(), IsNaive);
    std::string all_names;
    for(const auto& s : solvers)
        all_names += (all_names.empty() ? "" : ", ") + s;
    EXPECT_TRUE(any_naive)
        << "Expected at least one naive solver in results with MIOPEN_NAIVE_TIMEOUT=0" << "; got: ["
        << all_names << "]";
}

// With MIOPEN_NAIVE_TIMEOUT=1 and MIOPEN_DEBUG_FIND_ONLY_SOLVER restricting the candidate
// set to ConvDirectNaiveConvFwd, there are no non-naive solutions so defer_naive=false and
// the timeout cannot fire (path B). Find must return a naive result rather than nothing.
TEST(GPU_NaiveTimeout_FP32, NaiveNotSuppressedWhenOnlySolver)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<std::string> only_naive(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                              std::string{"ConvDirectNaiveConvFwd"});
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty())
        << "miopenFindSolutions returned no results: naive must not be suppressed when it is the "
           "only applicable solver (path B)";
    const bool all_naive = std::all_of(solvers.begin(), solvers.end(), IsNaive);
    EXPECT_TRUE(all_naive)
        << "Expected only naive solvers when MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd";
}

// With MIOPEN_NAIVE_TIMEOUT=1 (default), find must return results. The naive solver may or
// may not appear depending on whether it finished within the 3x budget — both outcomes are
// valid. This test only verifies that the timeout mechanism does not break the find path.
TEST(GPU_NaiveTimeout_FP32, FindSucceedsWithTimeoutEnabled)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty()) << "miopenFindSolutions returned no results with timeout enabled";
}

// With MIOPEN_NAIVE_TIMEOUT_FACTOR=1 (0.01x budget), the wall-clock deadline expires
// before the naive kernel can finish. Naive must be absent from the results.
TEST(GPU_NaiveTimeout_FP32, NaiveSkippedWithTinyBudget)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 1);
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty()) << "miopenFindSolutions returned no results";
    const bool any_naive = std::any_of(solvers.begin(), solvers.end(), IsNaive);
    std::string all_names;
    for(const auto& s : solvers)
        all_names += (all_names.empty() ? "" : ", ") + s;
    EXPECT_FALSE(any_naive) << "Naive solver should have been short-circuited with 1% budget"
                            << "; got: [" << all_names << "]";
}

// CPU-only: verify that MIOPEN_NAIVE_TIMEOUT can be set and read back via the debug
// registry (the same path ScopedEnvironment uses in the GPU tests).
TEST(CPU_NaiveTimeout_NONE, EnvVarRoundTrips)
{
    {
        ScopedEnvironment<bool> guard(MIOPEN_NAIVE_TIMEOUT, true);
        EXPECT_TRUE(lib_env::value<bool>(MIOPEN_NAIVE_TIMEOUT));
    }
    {
        ScopedEnvironment<bool> guard(MIOPEN_NAIVE_TIMEOUT, false);
        EXPECT_FALSE(lib_env::value<bool>(MIOPEN_NAIVE_TIMEOUT));
    }
}

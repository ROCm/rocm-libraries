// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for MIOPEN_NAIVE_DISABLE_IF_ALT behavior in the find path.
//
// Paths through EvaluateInvokers naive-deferring logic (solver_finders.cpp):
//
//   Path  naive_disable  defer_naive  non_naive_succeeded  Outcome
//   ----  -------------  -----------  -------------------  -------------------------
//   A     false          -            -                    naive evaluated (flag off)
//   B     true           false        -                    naive evaluated (no non-naive exists)
//   C     true           true         false                naive evaluated (non-naive all failed)
//   D     true           true         true                 naive SKIPPED (intended fast path)
//
// Tests cover paths A, B, and D. Path C is not exercised:
//   Path C: requires fault injection into the invoker execution path at runtime.
//
// Path B uses MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd to restrict the candidate
// set to the naive solver only. GetEnvFindOnlySolver() re-evaluates the env var on every call
// under MIOPEN_BUILD_TESTING (no static cache) so ScopedEnvironment takes effect.
//
// Tests assert presence/absence of naive solvers rather than which solver wins by timing,
// so results are not sensitive to GPU performance variation.
//
// Each GPU test uses ScopedFindDb to disable miopen::debug::testing_find_db_enabled for the
// duration of the test, ensuring TryLoad always calls the regenerator (FindCore / EvaluateInvokers)
// rather than returning a cached find DB entry.

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/solver_id.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../tensor_holder.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_DISABLE_IF_ALT)

namespace {

// RAII: disables the user find DB for the duration of the test so FindCore /
// EvaluateInvokers always runs rather than returning a cached result.
// miopen::debug::testing_find_db_enabled is declared in <miopen/find_db.hpp> and
// checked in TryLoad: when false, construct_db() returns null → in_sync=false →
// regenerator() (FindCore) is always called. Saves and restores the previous value
// so tests that run before or after us with find DB disabled are not disturbed.
struct ScopedFindDb
{
    ScopedFindDb() : prev_(miopen::debug::testing_find_db_enabled)
    {
        miopen::debug::testing_find_db_enabled = false;
    }
    ~ScopedFindDb() { miopen::debug::testing_find_db_enabled = prev_; }

    bool prev_;
};

// Returns solver names (fastest first) from miopenFindSolutions for a small NCHW FP32 forward
// convolution: 1×64×14×14 input, 64×3×3 filter, pad=1, stride=1.
// This shape attracts ImplicitGEMM/Winograd on all supported GPUs, ensuring a non-naive winner.
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

} // namespace

// With MIOPEN_NAIVE_DISABLE_IF_ALT=1, no naive solver should appear anywhere in the results
// for a shape that has non-naive alternatives.
TEST(GPU_NaiveDisableIfAlt_FP32, NaiveAbsentFromAllResultsWhenDisabled)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_cache; // disables find DB so FindCore / EvaluateInvokers always runs
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_DISABLE_IF_ALT, true);
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty()) << "miopenFindSolutions returned no results";
    for(const auto& s : solvers)
    {
        EXPECT_FALSE(IsNaive(s))
            << "Naive solver found in results with MIOPEN_NAIVE_DISABLE_IF_ALT=1: " << s;
    }
}

// With MIOPEN_NAIVE_DISABLE_IF_ALT=0 (opt-out), a naive solver must appear somewhere in the
// results, confirming that the skip logic in EvaluateInvokers did not fire.
// This tests the opt-out code path directly without depending on which solver wins by timing.
TEST(GPU_NaiveDisableIfAlt_FP32, NaivePresentInResultsWhenAllowed)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_cache; // disables find DB so FindCore / EvaluateInvokers always runs
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_DISABLE_IF_ALT, false);
    auto solvers = RunFind(handle);

    ASSERT_FALSE(solvers.empty()) << "miopenFindSolutions returned no results";
    const bool any_naive = std::any_of(solvers.begin(), solvers.end(), IsNaive);
    std::string all_names;
    for(const auto& s : solvers)
        all_names += (all_names.empty() ? "" : ", ") + s;
    EXPECT_TRUE(any_naive)
        << "Expected at least one naive solver in results with MIOPEN_NAIVE_DISABLE_IF_ALT=0"
        << "; got: [" << all_names << "]";
}

// With MIOPEN_NAIVE_DISABLE_IF_ALT=1 and MIOPEN_DEBUG_FIND_ONLY_SOLVER restricting the candidate
// set to ConvDirectNaiveConvFwd, there are no non-naive solutions so defer_naive=false and the
// skip guard cannot fire (path B). Find must return a naive result rather than nothing.
TEST(GPU_NaiveDisableIfAlt_FP32, NaiveNotSuppressedWhenOnlySolver)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_DISABLE_IF_ALT, true);
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

// CPU-only: verify that MIOPEN_NAIVE_DISABLE_IF_ALT can be set and read back via the debug
// registry (the same path ScopedEnvironment uses in the GPU tests).
TEST(CPU_NaiveDisableIfAlt_NONE, EnvVarRoundTrips)
{
    {
        ScopedEnvironment<bool> guard(MIOPEN_NAIVE_DISABLE_IF_ALT, true);
        EXPECT_TRUE(lib_env::value<bool>(MIOPEN_NAIVE_DISABLE_IF_ALT));
    }
    {
        ScopedEnvironment<bool> guard(MIOPEN_NAIVE_DISABLE_IF_ALT, false);
        EXPECT_FALSE(lib_env::value<bool>(MIOPEN_NAIVE_DISABLE_IF_ALT));
    }
}

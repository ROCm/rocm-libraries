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
//   D     true           true         true                 naive skipped by the run time bound,
//                                                          or timed: may pass or be skipped
//
// Find evaluates one algorithm at a time, so defer_naive looks at the solvers of the algorithm
// being evaluated, plus whatever succeeded in the algorithms before it. Path C therefore needs
// a non-naive Direct solver that fails next to naive. It is reached without fault injection where
// ConvHipConv is a candidate (fp16 NCHW, gfx950): Find runs without a workspace and ConvHipConv
// needs one for its transposes. Where it is not a candidate that test skips and path C is not
// covered. Path B is also reached with a failing solver of another algorithm (GemmFwdRest, in the
// GEMM algorithm).
//
// CaptureAfterATimedOutNaiveEvaluationSucceeds covers issue #12121: a naive
// evaluation abandoned by find must not break a later HIP graph capture.
//
// A naive solver whose run time lower bound (conv::NaiveTimeFloorMs) exceeds the budget is
// skipped before it is built or launched, so nothing is abandoned. MIOPEN_NAIVE_BOUND_SKIP=0
// turns that off; CaptureAfterATimedOutNaiveEvaluationSucceeds sets it so a kernel is abandoned.
//
// Path B uses MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd to restrict
// the candidate set to the naive solver only.
//
// Tests assert presence/absence of naive solvers rather than which solver wins
// by timing, so results are not sensitive to GPU performance variation.
//
// Each GPU test uses ScopedFindDb to disable miopen::debug::testing_find_db_enabled
// so TryLoad always calls the regenerator (FindCore / EvaluateInvokers).

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solver_finders.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../tensor_holder.hpp"
#include "find_db.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_TIMEOUT)
MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_TIMEOUT_FACTOR)
MIOPEN_LIB_ENV_VAR(MIOPEN_NAIVE_BOUND_SKIP)

namespace {

using miopen::ScopedFindDb;

template <class T = float>
std::vector<std::string> RunFind(miopenHandle_t handle, size_t max_solutions = 8)
{
    tensor<T> x{1, 64, 14, 14};
    tensor<T> w{64, 64, 3, 3};

    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    tensor<T> y{conv.GetForwardOutputTensor(x.desc, w.desc)};

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
// ConvBinWinograd3x3U accepts, and that solver is applicable only on these
// archs.  There IsWinograd3x3SupportedAndFast() sets use_winograd_only, which
// disables the Direct finder outright, so ConvDirectNaive is never a candidate
// and assertions about its presence or absence say nothing about the timeout.
bool IsWinogradOnlyArch(const miopen::Handle& handle)
{
    const auto name = handle.GetDeviceName();
    return name == "gfx900" || name == "gfx906" || name == "gfx908";
}

std::string Join(const std::vector<std::string>& names)
{
    std::string all_names;
    for(const auto& s : names)
        all_names += (all_names.empty() ? "" : ", ") + s;
    return all_names;
}

struct FoundSolvers
{
    std::vector<std::string> solvers;
    std::string log; // what the library logged at Info level during the Find
};

// RunFind with the library's Info log captured: it records which naive solvers are evaluated and
// why one is skipped.
template <class T = float>
FoundSolvers
RunFindLogged(miopenHandle_t handle, int log_level_value = 5, size_t max_solutions = 64)
{
    FoundSolvers found;
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, log_level_value);
    testing::internal::CaptureStderr();
    found.solvers = RunFind<T>(handle, max_solutions);
    found.log     = testing::internal::GetCapturedStderr();
    return found;
}

// EvaluateInvokers() logged that the run time bound alone dropped a naive solver.
bool LogShowsBoundSkip(const std::string& log)
{
    return log.find("run time lower bound") != std::string::npos;
}

// EvaluateInvokers() started to evaluate a naive solver, which builds and launches it.
bool LogShowsNaiveEvaluation(const std::string& log)
{
    std::istringstream lines{log};
    for(std::string line; std::getline(lines, line);)
        if(line.find("Evaluating Solver:") != std::string::npos &&
           line.find("Naive") != std::string::npos)
            return true;
    return false;
}

// A naive kernel file was named before the first solver evaluation. The precompile step logs every
// kernel it builds at the Info2 level, whatever the kernel cache, so this tells whether naive was
// in it.
bool LogShowsNaiveKernelBeforeFirstEvaluation(const std::string& log)
{
    std::istringstream lines{log};
    for(std::string line; std::getline(lines, line);)
    {
        if(line.find("Evaluating Solver:") != std::string::npos)
            return false;
        if(line.find("naive_conv") != std::string::npos)
            return true;
    }
    return false;
}

/// Device buffers for a forward convolution, allocated outside any capture.
struct ConvBuffers
{
    void* x  = nullptr;
    void* w  = nullptr;
    void* y  = nullptr;
    void* ws = nullptr;

    ~ConvBuffers()
    {
        for(void* p : {x, w, y, ws})
            (void)hipFree(p);
    }
};

struct LegacyFind
{
    miopenStatus_t status = miopenStatusUnknownError;
    int returned          = 0;
    miopenConvAlgoPerf_t perf[8]{};
    std::string log; // what the library logged at Info level during the Find
};

// The legacy Find entry point takes the caller's workspace, and here it is null. A solver that
// needs a workspace is rejected for lack of one. (miopenFindSolutions provides its own workspace
// and cannot reach this.) The convolution is the one RunFind uses, in the default layout.
template <class T>
void RunLegacyFindWithoutWorkspace(miopenHandle_t handle, LegacyFind& found)
{
    tensor<T> x{1, 64, 14, 14};
    tensor<T> w{64, 64, 3, 3};
    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    tensor<T> y{conv.GetForwardOutputTensor(x.desc, w.desc)};

    ConvBuffers buffers;
    ASSERT_EQ(hipMalloc(&buffers.x, x.desc.GetNumBytes()), hipSuccess);
    ASSERT_EQ(hipMalloc(&buffers.w, w.desc.GetNumBytes()), hipSuccess);
    ASSERT_EQ(hipMalloc(&buffers.y, y.desc.GetNumBytes()), hipSuccess);
    ASSERT_EQ(hipMemset(buffers.x, 0, x.desc.GetNumBytes()), hipSuccess);
    ASSERT_EQ(hipMemset(buffers.w, 0, w.desc.GetNumBytes()), hipSuccess);

    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 5);
    testing::internal::CaptureStderr();
    found.status = miopenFindConvolutionForwardAlgorithm(handle,
                                                         &x.desc,
                                                         buffers.x,
                                                         &w.desc,
                                                         buffers.w,
                                                         &conv,
                                                         &y.desc,
                                                         buffers.y,
                                                         8,
                                                         &found.returned,
                                                         found.perf,
                                                         nullptr,
                                                         0,
                                                         false);
    found.log    = testing::internal::GetCapturedStderr();
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

// With MIOPEN_NAIVE_TIMEOUT_FACTOR=1 (0.01x budget) the run time lower bound of the naive solver
// is far above its budget, so Find skips it before building or launching it. Naive must be absent
// from the results, and the log must show that the bound made the decision: no naive solver was
// evaluated and the skip is reported as a bound skip, not as a timeout.
TEST(GPU_NaiveTimeout_FP32, NaiveSkippedWithTinyBudget)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, true);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 1);
    const auto found = RunFindLogged(handle);

    ASSERT_FALSE(found.solvers.empty()) << "miopenFindSolutions returned no results";
    const bool any_naive = std::any_of(found.solvers.begin(), found.solvers.end(), IsNaive);
    EXPECT_FALSE(any_naive) << "Naive solver should have been skipped with 1% budget" << "; got: ["
                            << Join(found.solvers) << "]";
    EXPECT_TRUE(LogShowsBoundSkip(found.log)) << "the skip did not come from the run time bound";
    EXPECT_FALSE(LogShowsNaiveEvaluation(found.log))
        << "a naive solver was built and launched although its bound exceeds the budget";
}

// With a budget of 1000x the best non-naive time the bound is far below it. Find must not skip the
// naive solver: it is evaluated and returned. This fails if the skip fires whenever a non-naive
// solver succeeded, or if the comparison of bound and budget is the wrong way round.
TEST(GPU_NaiveTimeout_FP32, NaiveEvaluatedWhenBoundIsBelowBudget)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, true);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 100000);
    const auto found = RunFindLogged(handle);

    ASSERT_FALSE(found.solvers.empty()) << "miopenFindSolutions returned no results";
    EXPECT_FALSE(LogShowsBoundSkip(found.log)) << "naive was skipped although its bound is far "
                                                  "below the budget";
    EXPECT_TRUE(LogShowsNaiveEvaluation(found.log));
    const bool any_naive = std::any_of(found.solvers.begin(), found.solvers.end(), IsNaive);
    EXPECT_TRUE(any_naive) << "Expected the naive solver in the results with a 1000x budget"
                           << "; got: [" << Join(found.solvers) << "]";
}

// MIOPEN_NAIVE_BOUND_SKIP=0 restores the behavior without the bound: with a 1% budget the naive
// solver is built and launched, and the timeout abandons it.
TEST(GPU_NaiveTimeout_FP32, NaiveTimedOutWhenBoundSkipDisabled)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, false);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 1);
    const auto found = RunFindLogged(handle);

    ASSERT_FALSE(found.solvers.empty()) << "miopenFindSolutions returned no results";
    EXPECT_FALSE(LogShowsBoundSkip(found.log));
    EXPECT_TRUE(LogShowsNaiveEvaluation(found.log))
        << "with the bound off the naive solver must be built and launched";
    const bool any_naive = std::any_of(found.solvers.begin(), found.solvers.end(), IsNaive);
    EXPECT_FALSE(any_naive) << "the timeout should have abandoned naive with 1% budget"
                            << "; got: [" << Join(found.solvers) << "]";
}

// The precompile step builds the kernels of every solver in the list in parallel before the
// evaluation starts. Naive stays out of it while the bound can still skip it. These two tests read
// the precompiled kernels in the Info2 log. A new handle keeps earlier tests from hiding them
// through the handle's program cache, even when tests run in a different order.
TEST(GPU_NaiveTimeout_FP32, NaiveNotPrecompiledWhenBoundCanSkipIt)
{
    miopen::Handle handle_ref;
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, true);
    const auto found = RunFindLogged<bfloat16>(handle, 6);

    const bool any_non_naive = std::any_of(
        found.solvers.begin(), found.solvers.end(), [](const auto& s) { return !IsNaive(s); });
    if(!any_non_naive)
        GTEST_SKIP() << "no non-naive solver for this problem on " << handle_ref.GetDeviceName();

    EXPECT_FALSE(LogShowsNaiveKernelBeforeFirstEvaluation(found.log))
        << "naive was in the precompile although its bound can skip it";
}

// MIOPEN_NAIVE_BOUND_SKIP=0 puts naive back into the precompile.
TEST(GPU_NaiveTimeout_FP32, NaivePrecompiledWhenBoundSkipDisabled)
{
    miopen::Handle handle_ref;
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, false);
    const auto found = RunFindLogged<half_float::half>(handle, 6);

    const bool any_non_naive = std::any_of(
        found.solvers.begin(), found.solvers.end(), [](const auto& s) { return !IsNaive(s); });
    if(!any_non_naive)
        GTEST_SKIP() << "no non-naive solver for this problem on " << handle_ref.GetDeviceName();

    EXPECT_TRUE(LogShowsNaiveKernelBeforeFirstEvaluation(found.log))
        << "the naive kernel was not looked up before the first evaluation";
}

// Path B again, with a failing non-naive solver around. Find evaluates one algorithm at a time, and
// GemmFwdRest belongs to the GEMM algorithm, so the Direct algorithm holds naive alone. Find runs
// without a workspace, so GemmFwdRest is rejected for lack of one and no non-naive solver
// succeeds. Naive is then evaluated even though its bound is above its 1% budget: the bound and
// the timeout only apply once a non-naive solver has succeeded.
TEST(GPU_NaiveTimeout_FP32, NaiveEvaluatedWhenOtherAlgorithmsNeedWorkspace)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, true);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 1);
    ScopedEnvironment<std::string> only(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                        std::string{"GemmFwdRest;ConvDirectNaiveConvFwd"});

    LegacyFind found;
    ASSERT_NO_FATAL_FAILURE(RunLegacyFindWithoutWorkspace<float>(handle, found));

    if(found.log.find("Solver <GemmFwdRest>, workspace required") == std::string::npos)
        GTEST_SKIP() << "GemmFwdRest is not a candidate on " << handle_ref.GetDeviceName();

    ASSERT_EQ(found.status, miopenStatusSuccess);
    ASSERT_GE(found.returned, 1) << "naive must be evaluated when no non-naive solver succeeded";
    for(int i = 0; i < found.returned; ++i)
        EXPECT_EQ(found.perf[i].fwd_algo, miopenConvolutionFwdAlgoDirect) << "result " << i;
    EXPECT_TRUE(LogShowsNaiveEvaluation(found.log));
    EXPECT_FALSE(LogShowsBoundSkip(found.log));
}

// Path C. The Direct algorithm holds a non-naive solver next to naive, and that solver fails, so
// no non-naive solver has succeeded when naive comes up. ConvHipConv stages an fp16 NCHW problem
// through transposes in the workspace, and Find runs without one, so it is rejected for lack of a
// workspace. Naive must then be evaluated and returned: the bound and the timeout apply only
// once a non-naive solver has succeeded, and until then there is no best time to measure against.
// The test skips where ConvHipConv is not a candidate for the problem (not gfx950 for example).
// A zero timeout budget makes an accidental cutoff when no non-naive solver succeeded
// deterministically drop naive, instead of relying on overflow from FLT_MAX.
TEST(GPU_NaiveTimeout_FP32, NaiveEvaluatedWhenEveryNonNaiveSolverOfItsAlgorithmFails)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    if(IsWinogradOnlyArch(handle_ref))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << handle_ref.GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, true);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 0);
    ScopedEnvironment<std::string> only(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                        std::string{"ConvHipConv;ConvDirectNaiveConvFwd"});

    LegacyFind found;
    ASSERT_NO_FATAL_FAILURE(RunLegacyFindWithoutWorkspace<half_float::half>(handle, found));

    // Where ConvHipConv is not a candidate the Direct algorithm holds naive alone (path B), so Find
    // still succeeds and evaluates naive. Check that first, so a broken setup fails here instead of
    // skipping.
    ASSERT_EQ(found.status, miopenStatusSuccess);
    ASSERT_GE(found.returned, 1) << "naive must be evaluated when every non-naive solver failed";
    ASSERT_TRUE(LogShowsNaiveEvaluation(found.log));

    if(found.log.find("[EvaluateInvokers] Solver <ConvHipConv>, workspace required") ==
       std::string::npos)
        GTEST_SKIP() << "ConvHipConv is not a candidate for this problem on "
                     << handle_ref.GetDeviceName() << ", path C is not reached";

    for(int i = 0; i < found.returned; ++i)
        EXPECT_EQ(found.perf[i].fwd_algo, miopenConvolutionFwdAlgoDirect) << "result " << i;
    EXPECT_FALSE(LogShowsBoundSkip(found.log));
    EXPECT_EQ(found.log.find("Skipped naive solver"), std::string::npos)
        << "naive was dropped although no non-naive solver had succeeded";
}

// Regression for issue #12121. With a 1% budget the naive evaluation is
// abandoned - the Direct finder runs last, so just before find returns - and
// the convolution below is the first launch on the handle after that, made
// inside a capture in each mode. The capture must survive whatever the
// abandoned evaluation left behind. There is deliberately no uncaptured run in
// between: one would have reclaimed a finished evaluation before the capture
// and hidden a regression. Find has already loaded the solution's kernels.
TEST(GPU_NaiveTimeout_FP32, CaptureAfterATimedOutNaiveEvaluationSucceeds)
{
    if(IsWinogradOnlyArch(get_handle()))
        GTEST_SKIP() << "Direct finder is disabled by use_winograd_only on "
                     << get_handle().GetDeviceName();

    ScopedFindDb no_cache;
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);
    // The run-time bound would skip naive before it is launched, and nothing would be abandoned.
    ScopedEnvironment<bool> guard_bound(MIOPEN_NAIVE_BOUND_SKIP, false);
    ScopedEnvironment<int> guard_factor(MIOPEN_NAIVE_TIMEOUT_FACTOR, 1);

    tensor<float> x{8, 128, 28, 28};
    tensor<float> w{128, 128, 3, 3};
    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    tensor<float> y{conv.GetForwardOutputTensor(x.desc, w.desc)};

    for(const auto mode : {hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal})
    {
        SCOPED_TRACE(mode == hipStreamCaptureModeGlobal ? "global capture mode"
                                                        : "thread-local capture mode");

        // A handle on its own stream: the default handle's null stream cannot be
        // captured.
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        miopenHandle_t handle = nullptr;
        ASSERT_EQ(miopenCreateWithStream(&handle, stream), miopenStatusSuccess);

        miopenProblem_t problem = nullptr;
        ASSERT_EQ(miopenCreateConvProblem(&problem, &conv, miopenProblemDirectionForward),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionW, &w.desc),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionY, &y.desc),
                  miopenStatusSuccess);

        std::vector<miopenSolution_t> solutions(8);
        size_t found = 0;
        ASSERT_EQ(miopenFindSolutions(
                      handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                  miopenStatusSuccess);
        solutions.resize(found);
        ASSERT_FALSE(solutions.empty()) << "miopenFindSolutions returned no results";
        for(auto sol : solutions)
        {
            uint64_t solver_id = 0;
            ASSERT_EQ(miopenGetSolutionSolverId(sol, &solver_id), miopenStatusSuccess);
            ASSERT_FALSE(IsNaive(miopen::solver::Id{solver_id}.ToString()))
                << "the naive evaluation was not cut short, so nothing was abandoned";
        }

        const auto best = solutions.front();
        ConvBuffers buffers;
        size_t ws_size = 0;
        ASSERT_EQ(miopenGetSolutionWorkspaceSize(best, &ws_size), miopenStatusSuccess);
        ASSERT_EQ(hipMalloc(&buffers.x, x.desc.GetNumBytes()), hipSuccess);
        ASSERT_EQ(hipMalloc(&buffers.w, w.desc.GetNumBytes()), hipSuccess);
        ASSERT_EQ(hipMalloc(&buffers.y, y.desc.GetNumBytes()), hipSuccess);
        if(ws_size > 0)
            ASSERT_EQ(hipMalloc(&buffers.ws, ws_size), hipSuccess);
        const miopenTensorArgument_t args[] = {
            {miopenTensorConvolutionX, nullptr, buffers.x},
            {miopenTensorConvolutionW, nullptr, buffers.w},
            {miopenTensorConvolutionY, nullptr, buffers.y},
        };

        ASSERT_EQ(hipStreamBeginCapture(stream, mode), hipSuccess);
        const auto run_status = miopenRunSolution(handle, best, 3, args, buffers.ws, ws_size);
        hipGraph_t graph      = nullptr;
        const auto end_status = hipStreamEndCapture(stream, &graph);

        EXPECT_EQ(run_status, miopenStatusSuccess) << "the captured convolution failed";
        EXPECT_EQ(end_status, hipSuccess) << "the capture was invalidated";
        EXPECT_NE(graph, nullptr);

        if(graph != nullptr)
            (void)hipGraphDestroy(graph);
        for(auto sol : solutions)
            miopenDestroySolution(sol);
        miopenDestroyProblem(problem);
        (void)hipStreamSynchronize(stream); // cleanup; fails after an invalidated capture
        miopenDestroy(handle);              // waits for the abandoned naive kernel
        (void)hipStreamDestroy(stream);
    }
}

namespace {

struct FloorCase
{
    const char* name;
    std::size_t n, c, k, h, w, y, x;
    int pad, stride, dilation, groups;
    bool fp16_nhwc;
};

// Padding-dominated shapes come first: a filter far larger than the input, where most taps read
// padding that the naive kernels skip. The first of them is a 1x1 input with a 7x7 filter and
// padding 3, the shape for which the floor used to exceed the naive kernel's run time.
const FloorCase floor_cases[] = {
    {"1x1 input, 7x7 filter, pad 3, fp16 NHWC", 256, 256, 256, 1, 1, 7, 7, 3, 1, 1, 1, true},
    {"1x1 input, 11x11 filter, pad 5", 256, 128, 128, 1, 1, 11, 11, 5, 1, 1, 1, false},
    {"2x2 input, 7x7 filter, pad 3", 128, 256, 256, 2, 2, 7, 7, 3, 1, 1, 1, false},
    {"7x7 input, dilation 4, 3x3 filter, pad 4", 64, 256, 256, 7, 7, 3, 3, 4, 1, 4, 1, false},
    {"stride 2, 5x5 filter, pad 2, 4 groups", 16, 64, 128, 15, 15, 5, 5, 2, 2, 1, 4, false},
    {"depthwise, 3x3 filter, pad 1", 8, 64, 64, 14, 14, 3, 3, 1, 1, 1, 64, false},
    {"3x3 filter, pad 1", 8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, false},
    {"1x1 filter, stride 2", 8, 128, 256, 56, 56, 1, 1, 0, 2, 1, 1, false},
};

miopen::conv::ProblemDescription MakeFloorProblem(const FloorCase& fc,
                                                  miopen::conv::Direction direction)
{
    const auto type   = fc.fp16_nhwc ? miopenHalf : miopenFloat;
    const auto layout = fc.fp16_nhwc ? miopenTensorNHWC : miopenTensorNCHW;
    const miopen::TensorDescriptor x{
        type, layout, std::vector<std::size_t>{fc.n, fc.c, fc.h, fc.w}};
    const miopen::TensorDescriptor w{
        type, layout, std::vector<std::size_t>{fc.k, fc.c / fc.groups, fc.y, fc.x}};
    const miopen::ConvolutionDescriptor conv{2,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {fc.pad, fc.pad},
                                             {fc.stride, fc.stride},
                                             {fc.dilation, fc.dilation},
                                             {0, 0},
                                             fc.groups};
    const auto y =
        conv.GetForwardOutputTensorWithLayout(x, w, fc.fp16_nhwc ? "NHWC" : "NCHW", type);
    // The tensors of a backward problem are given in the order dy, w, dx.
    if(direction == miopen::conv::Direction::Forward)
        return {x, w, y, conv, direction};
    return {y, w, x, conv, direction};
}

float FloorMs(const FloorCase& fc, miopen::conv::Direction direction)
{
    return miopen::conv::NaiveTimeFloorMs(get_handle(), MakeFloorProblem(fc, direction));
}

// Number of (output position, filter tap) pairs along one dimension whose tap reads inside the
// input, counted one by one.
std::size_t TapsInInput(int out, int filter, int in, int pad, int stride, int dilation)
{
    std::size_t taps = 0;
    for(int o = 0; o < out; ++o)
        for(int t = 0; t < filter; ++t)
        {
            const int pos = stride * o - pad + dilation * t;
            taps += (pos >= 0 && pos < in) ? 1 : 0;
        }
    return taps;
}

int OutSize(int in, int filter, int pad, int stride, int dilation)
{
    return (in + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
}

} // namespace

// The bound EvaluateInvokers() uses to skip a naive solver before launching it must be positive
// and must grow with the amount of work in the problem.
TEST(GPU_NaiveTimeout_FP32, NaiveTimeFloorScalesWithWork)
{
    auto& handle = get_handle();

    const auto floor_ms = [&](int batch) {
        tensor<float> x{static_cast<size_t>(batch), 128, 28, 28};
        tensor<float> w{128, 128, 3, 3};
        miopen::ConvolutionDescriptor conv{
            2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
        tensor<float> y{conv.GetForwardOutputTensor(x.desc, w.desc)};
        const miopen::conv::ProblemDescription problem{
            x.desc, w.desc, y.desc, conv, miopen::conv::Direction::Forward};
        return miopen::conv::NaiveTimeFloorMs(handle, problem);
    };

    const float small = floor_ms(8);
    const float large = floor_ms(16);
    ASSERT_GT(small, 0.0f);
    EXPECT_NEAR(large / small, 2.0f, 1e-3f);
}

// The naive kernels do nothing for a filter tap that reads padding, so a floor that counted such
// taps could exceed the kernel's run time and drop a naive solver that would have won. The floor
// is the floor of the same problem with a 1x1 filter times the number of taps that read inside
// the input, which the test counts one by one.
TEST(GPU_NaiveTimeout_FP32, NaiveTimeFloorCountsOnlyTapsInInput)
{
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);

    for(const auto& fc : floor_cases)
    {
        SCOPED_TRACE(fc.name);
        const int ho    = OutSize(fc.h, fc.y, fc.pad, fc.stride, fc.dilation);
        const int wo    = OutSize(fc.w, fc.x, fc.pad, fc.stride, fc.dilation);
        const auto taps = TapsInInput(ho, fc.y, fc.h, fc.pad, fc.stride, fc.dilation) *
                          TapsInInput(wo, fc.x, fc.w, fc.pad, fc.stride, fc.dilation);

        // One multiply-accumulate per output pixel: a 1x1 filter on a 1x1 image.
        FloorCase unit = fc;
        unit.h = unit.w = unit.y = unit.x = 1;
        unit.pad                          = 0;
        unit.stride = unit.dilation = 1;
        const float unit_floor      = FloorMs(unit, miopen::conv::Direction::Forward);
        ASSERT_GT(unit_floor, 0.0f);

        const float got      = FloorMs(fc, miopen::conv::Direction::Forward);
        const float expected = unit_floor * static_cast<float>(taps);
        EXPECT_NEAR(got / expected, 1.0f, 1e-4f) << "taps in input " << taps << ", all taps "
                                                 << static_cast<std::size_t>(ho) * wo * fc.y * fc.x;
    }
}

// Each output channel of a grouped convolution sums over C/G input channels, so the floor is the
// floor of the same problem with one group divided by the group count. Without the divisor the
// floor of a depthwise convolution would be G times too large and exceed the kernel's run time.
TEST(GPU_NaiveTimeout_FP32, NaiveTimeFloorDividesByTheGroupCount)
{
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);

    bool any_grouped = false;
    for(const auto& fc : floor_cases)
    {
        if(fc.groups == 1)
            continue;
        any_grouped = true;
        SCOPED_TRACE(fc.name);

        FloorCase one_group = fc;
        one_group.groups    = 1;
        const float single  = FloorMs(one_group, miopen::conv::Direction::Forward);
        ASSERT_GT(single, 0.0f);

        const float expected = single / static_cast<float>(fc.groups);
        EXPECT_NEAR(FloorMs(fc, miopen::conv::Direction::Forward) / expected, 1.0f, 1e-4f)
            << "groups " << fc.groups;
    }
    EXPECT_TRUE(any_grouped);
}

// Forward, backward data and backward weights of one convolution do the same work in the naive
// kernels, so they get the same floor.
TEST(GPU_NaiveTimeout_FP32, NaiveTimeFloorIsTheSameInEveryDirection)
{
    ScopedEnvironment<bool> guard_naive(MIOPEN_NAIVE_TIMEOUT, true);

    for(const auto& fc : floor_cases)
    {
        SCOPED_TRACE(fc.name);
        const float fwd = FloorMs(fc, miopen::conv::Direction::Forward);
        const float bwd = FloorMs(fc, miopen::conv::Direction::BackwardData);
        const float wrw = FloorMs(fc, miopen::conv::Direction::BackwardWeights);
        ASSERT_GT(fwd, 0.0f);
        EXPECT_EQ(bwd, fwd) << "backward data";
        EXPECT_EQ(wrw, fwd) << "backward weights";
    }
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

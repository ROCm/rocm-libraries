// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// miopenConvolution*GetWorkSpaceSize must report the GEMM workspace of a dilated convolution when
// GEMM is the only alternative to the naive solvers and its workspace fits in 85% of the device
// memory free at query time, as miopen.h documents (GetWorkSpaceSizeGEMM in src/convolution.cpp),
// in every direction, and must leave non-dilated problems unchanged.
//
// Applications such as PyTorch size the Find workspace from that query. When it under-reports,
// Find drops the GEMM solvers for lack of workspace, and if no other solver applies it fails with
// "No suitable algorithm was found" (rocm-libraries#14965). A size that does not fit in free
// memory is not reported, so callers do not try to allocate it (#6974).
//
// MIOPEN_DEBUG_FIND_ONLY_SOLVER makes one GEMM solver the only candidate on every arch, so GEMM is
// the alternative to the naive solvers, and the workspace queries also disable the other solver
// families, so the reported size is that solver's. Free memory comes from AvailableMemoryHandle.
// ScopedFindDb keeps find-db records out, except in the find-db test, which points the find-db at
// a temporary file.

#include <gtest/gtest.h>

#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../tensor_holder.hpp"
#include "../workspace.hpp"
#include "find_db.hpp"

#include <cmath>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using miopen::ScopedFindDb;

// A real device handle that reports a fixed amount of free memory.
class AvailableMemoryHandle final : public miopen::Handle
{
public:
    explicit AvailableMemoryHandle(std::size_t available) : available_(available) {}
    std::size_t GetAvailableMemory() const override { return available_; }

private:
    std::size_t available_;
};

// The share of free memory miopen.h documents for a dilated GEMM workspace.
constexpr double DocumentedFreeMemoryFraction = 0.85;

// Free memory whose budget is `margin` bytes above (or, negative, below) `required`.
std::size_t FreeMemoryForBudget(std::size_t required, double margin)
{
    return static_cast<std::size_t>(
        std::ceil((static_cast<double>(required) + margin) / DocumentedFreeMemoryFraction));
}

enum class Direction
{
    Forward,
    BackwardData,
    BackwardWeights,
};

// The GEMM solver a dilated problem of `direction` uses in these tests.
std::string GemmSolver(Direction direction)
{
    switch(direction)
    {
    case Direction::Forward: return "GemmFwdRest";
    case Direction::BackwardData: return "GemmBwdRest";
    case Direction::BackwardWeights: return "GemmWrwUniversal";
    }
    return {};
}

std::string DirectionName(Direction direction)
{
    switch(direction)
    {
    case Direction::Forward: return "Forward";
    case Direction::BackwardData: return "BackwardData";
    case Direction::BackwardWeights: return "BackwardWeights";
    }
    return {};
}

struct ConvProblem
{
    tensor<float> x;
    tensor<float> w;
    miopen::ConvolutionDescriptor conv;
    tensor<float> y;

    ConvProblem(std::vector<std::size_t> x_lens,
                std::vector<std::size_t> w_lens,
                std::vector<int> pads,
                std::vector<int> dilations)
        : x{x_lens},
          w{w_lens},
          conv{2, miopenConvolution, miopenPaddingDefault, pads, {1, 1}, dilations},
          y{conv.GetForwardOutputTensor(x.desc, w.desc)}
    {
    }

    // What miopenConvolution*GetWorkSpaceSize reports for `direction`.
    std::size_t QueriedWorkspace(miopenHandle_t handle, Direction direction)
    {
        std::size_t size = 0;
        miopenStatus_t status{};
        switch(direction)
        {
        case Direction::Forward:
            status = miopenConvolutionForwardGetWorkSpaceSize(
                handle, &w.desc, &x.desc, &conv, &y.desc, &size);
            break;
        case Direction::BackwardData:
            status = miopenConvolutionBackwardDataGetWorkSpaceSize(
                handle, &y.desc, &w.desc, &conv, &x.desc, &size);
            break;
        case Direction::BackwardWeights:
            status = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                handle, &y.desc, &x.desc, &conv, &w.desc, &size);
            break;
        }
        EXPECT_EQ(status, miopenStatusSuccess);
        return size;
    }

    // The workspace `solver` needs for `direction`.
    std::size_t
    SolverWorkspace(miopenHandle_t handle, Direction direction, const std::string& solver)
    {
        const auto id    = miopen::solver::Id{solver}.Value();
        std::size_t size = 0;
        miopenStatus_t status{};
        switch(direction)
        {
        case Direction::Forward:
            status = miopenConvolutionForwardGetSolutionWorkspaceSize(
                handle, &w.desc, &x.desc, &conv, &y.desc, id, &size);
            break;
        case Direction::BackwardData:
            status = miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
                handle, &y.desc, &w.desc, &conv, &x.desc, id, &size);
            break;
        case Direction::BackwardWeights:
            status = miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
                handle, &y.desc, &x.desc, &conv, &w.desc, id, &size);
            break;
        }
        EXPECT_EQ(status, miopenStatusSuccess) << solver;
        return size;
    }

    // Find (immediate API) with the given workspace; returns the chosen algorithm.
    miopenConvFwdAlgorithm_t FindForward(miopen::Handle& handle, std::size_t workspace_size)
    {
        auto x_dev = handle.Write(x.data);
        auto w_dev = handle.Write(w.data);
        auto y_dev = handle.Write(y.data);
        Workspace workspace{workspace_size};

        int returned_algo_count = 0;
        miopenConvAlgoPerf_t perf{};
        EXPECT_EQ(miopenFindConvolutionForwardAlgorithm(&handle,
                                                        &x.desc,
                                                        x_dev.get(),
                                                        &w.desc,
                                                        w_dev.get(),
                                                        &conv,
                                                        &y.desc,
                                                        y_dev.get(),
                                                        1,
                                                        &returned_algo_count,
                                                        &perf,
                                                        workspace.ptr(),
                                                        workspace.size(),
                                                        false),
                  miopenStatusSuccess);
        EXPECT_EQ(returned_algo_count, 1);
        return perf.fwd_algo;
    }

    // Forward Find 2.0 (miopenFindSolutions) with no preallocated workspace; returns each
    // solution's solver and workspace size.
    std::vector<std::pair<std::string, std::size_t>> FindSolutions(miopenHandle_t handle)
    {
        miopenProblem_t problem = nullptr;
        EXPECT_EQ(miopenCreateConvProblem(&problem, &conv, miopenProblemDirectionForward),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionW, &w.desc),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionY, &y.desc),
                  miopenStatusSuccess);

        constexpr std::size_t max_solutions = 8;
        std::vector<miopenSolution_t> solutions(max_solutions);
        std::size_t found = 0;
        // With no applicable solver that fits, Find reports an error and no solution.
        if(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, max_solutions) !=
           miopenStatusSuccess)
            found = 0;
        solutions.resize(found);

        std::vector<std::pair<std::string, std::size_t>> result;
        for(auto solution : solutions)
        {
            uint64_t solver_id = 0;
            std::size_t size   = 0;
            EXPECT_EQ(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
            EXPECT_EQ(miopenGetSolutionWorkspaceSize(solution, &size), miopenStatusSuccess);
            result.emplace_back(miopen::solver::Id{solver_id}.ToString(), size);
            miopenDestroySolution(solution);
        }
        miopenDestroyProblem(problem);
        return result;
    }

    // A forward solver, other than GEMM and the naive ones, that the immediate-mode API lists as
    // applicable, if any.
    std::optional<std::string> NonNaiveAlternativeToGemm(miopenHandle_t handle)
    {
        std::size_t count = 0;
        EXPECT_EQ(miopenConvolutionForwardGetSolutionCount(
                      handle, &w.desc, &x.desc, &conv, &y.desc, &count),
                  miopenStatusSuccess);
        std::vector<miopenConvSolution_t> solutions(count);
        EXPECT_EQ(miopenConvolutionForwardGetSolution(
                      handle, &w.desc, &x.desc, &conv, &y.desc, count, &count, solutions.data()),
                  miopenStatusSuccess);
        solutions.resize(count);
        for(const auto& solution : solutions)
        {
            const auto name = miopen::solver::Id{solution.solution_id}.ToString();
            if(solution.algorithm != miopenConvolutionAlgoGEMM &&
               name.find("Naive") == std::string::npos)
                return name;
        }
        return std::nullopt;
    }
};

// rocm-libraries#14965 in miniature: dilated fp32 Conv1d.
ConvProblem ReproDilatedConv() { return {{1, 32, 1, 1024}, {32, 32, 1, 3}, {0, 3}, {1, 3}}; }

// Im2Col: 256 * 3 * 3 * 256 * 256 * sizeof(float) = 576 MiB, dilated or not.
ConvProblem LargeConv(int dilation)
{
    return {{1, 256, 256, 256}, {16, 256, 3, 3}, {dilation, dilation}, {dilation, dilation}};
}

// `solver` is the only Find candidate, and the workspace query counts only the GEMM family.
struct OnlyGemm
{
    explicit OnlyGemm(const std::string& solver)
        : only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER, solver)
    {
    }

    ScopedEnvironment<std::string> only_solver;
    ScopedEnvironment<bool> no_direct{MIOPEN_DEBUG_CONV_DIRECT, false};
    ScopedEnvironment<bool> no_implicit_gemm{MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, false};
    ScopedEnvironment<bool> no_fft{MIOPEN_DEBUG_CONV_FFT, false};
    ScopedEnvironment<bool> no_winograd{MIOPEN_DEBUG_CONV_WINOGRAD, false};
};

// Margin around the budget boundary, well above double-to-size_t rounding.
constexpr double BoundaryMargin = 4096;

} // namespace

class GPU_ConvDilatedGemmWorkspaceBudget_FP32 : public ::testing::TestWithParam<Direction>
{
};

// Just inside the budget, the dilated GEMM workspace is reported.
TEST_P(GPU_ConvDilatedGemmWorkspaceBudget_FP32, WorkspaceWithinBudgetIsReported)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    const auto direction = GetParam();
    const auto solver    = GemmSolver(direction);
    ScopedFindDb no_find_db;
    OnlyGemm only_gemm{solver};

    auto problem        = LargeConv(2);
    const auto required = problem.SolverWorkspace(&get_handle(), direction, solver);
    ASSERT_GT(required, 0) << solver << " needs Im2Col workspace for this problem";

    AvailableMemoryHandle handle{FreeMemoryForBudget(required, BoundaryMargin)};
    EXPECT_EQ(problem.QueriedWorkspace(&handle, direction), required);
#endif
}

// Just outside the budget, the dilated GEMM workspace is not reported.
TEST_P(GPU_ConvDilatedGemmWorkspaceBudget_FP32, WorkspaceOverBudgetIsNotReported)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    const auto direction = GetParam();
    const auto solver    = GemmSolver(direction);
    ScopedFindDb no_find_db;
    OnlyGemm only_gemm{solver};

    auto problem        = LargeConv(2);
    const auto required = problem.SolverWorkspace(&get_handle(), direction, solver);
    ASSERT_GT(required, 0) << solver << " needs Im2Col workspace for this problem";

    AvailableMemoryHandle handle{FreeMemoryForBudget(required, -BoundaryMargin)};
    EXPECT_EQ(problem.QueriedWorkspace(&handle, direction), 0);
#endif
}

// The budget applies to dilated problems only: a non-dilated GEMM workspace is reported with
// half of it free, as before the budget.
TEST_P(GPU_ConvDilatedGemmWorkspaceBudget_FP32, NonDilatedWorkspaceIgnoresBudget)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    const auto direction = GetParam();
    const auto solver    = GemmSolver(direction);
    ScopedFindDb no_find_db;
    OnlyGemm only_gemm{solver};

    auto problem        = LargeConv(1);
    const auto required = problem.SolverWorkspace(&get_handle(), direction, solver);
    ASSERT_GT(required, 0) << solver << " needs Im2Col workspace for this problem";

    AvailableMemoryHandle handle{required / 2};
    EXPECT_EQ(problem.QueriedWorkspace(&handle, direction), required);
#endif
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDilatedGemmWorkspaceBudget_FP32,
                         testing::Values(Direction::Forward,
                                         Direction::BackwardData,
                                         Direction::BackwardWeights),
                         [](const auto& info) { return DirectionName(info.param); });

// Dilated GEMM is the alternative to the naive solvers only: where another solver applies, no GEMM
// workspace is reported, so Find does not benchmark GEMM as well. The query then matches the one
// with the GEMM family disabled.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, NoGemmWorkspaceWhereAnotherSolverApplies)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle = get_handle();
    ScopedFindDb no_find_db;

    auto problem = LargeConv(2);
    ASSERT_GT(problem.SolverWorkspace(&handle, Direction::Forward, "GemmFwdRest"), 0)
        << "GemmFwdRest needs Im2Col workspace for this problem";
    const auto alternative = problem.NonNaiveAlternativeToGemm(&handle);
    if(!alternative)
        GTEST_SKIP() << "Only GEMM and the naive solvers apply on this device";

    const auto queried = problem.QueriedWorkspace(&handle, Direction::Forward);
    ScopedEnvironment<bool> no_gemm(MIOPEN_DEBUG_CONV_GEMM, false);
    EXPECT_EQ(queried, problem.QueriedWorkspace(&handle, Direction::Forward))
        << *alternative << " applies";
#endif
}

// rocm-libraries#14965: Find given the queried workspace must select GemmFwdRest instead of
// failing.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindWithQueriedWorkspaceRunsGemmFwdRest)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle = get_handle();

    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem = ReproDilatedConv();

    const auto required = problem.SolverWorkspace(&handle, Direction::Forward, "GemmFwdRest");
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    const auto queried = problem.QueriedWorkspace(&handle, Direction::Forward);
    EXPECT_GE(queried, required);
    EXPECT_EQ(problem.FindForward(handle, queried), miopenConvolutionFwdAlgoGEMM);
#endif
}

// A find-db record of GemmFwdRest must not make the query report a workspace that does not fit in
// free memory: the record is honoured while it fits and skipped once it does not.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindDbRecordDoesNotBypassFreeMemoryBudget)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle = get_handle();

    const miopen::TempFile find_db{"miopen.test.conv_dilated_gemm_workspace"};
    const miopen::FindDbPathOverride find_db_override{find_db.Path()};
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem        = ReproDilatedConv();
    const auto required = problem.SolverWorkspace(&handle, Direction::Forward, "GemmFwdRest");
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";
    ASSERT_EQ(problem.FindForward(handle, required), miopenConvolutionFwdAlgoGEMM);
    std::ifstream records{find_db.Path()};
    const std::string text{std::istreambuf_iterator<char>{records}, {}};
    ASSERT_NE(text.find("GemmFwdRest"), std::string::npos) << "Find did not record GemmFwdRest";

    AvailableMemoryHandle roomy{2 * required};
    EXPECT_EQ(problem.QueriedWorkspace(&roomy, Direction::Forward), required);

    AvailableMemoryHandle tight{required / 2};
    EXPECT_EQ(problem.QueriedWorkspace(&tight, Direction::Forward), 0);
#endif
}

// In Fast mode the query takes the immediate-mode fallback's pick; a GEMM pick must not bypass
// the budget either.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FastModeFallbackDoesNotBypassFreeMemoryBudget)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem = ReproDilatedConv();
    // Set on the descriptor: the find mode is read from MIOPEN_FIND_MODE once per process.
    problem.conv.findMode.Set(miopen::FindMode::Values::Fast);
    const auto required = problem.SolverWorkspace(&get_handle(), Direction::Forward, "GemmFwdRest");
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    AvailableMemoryHandle roomy{2 * required};
    ASSERT_EQ(problem.QueriedWorkspace(&roomy, Direction::Forward), required);

    AvailableMemoryHandle tight{required / 2};
    EXPECT_EQ(problem.QueriedWorkspace(&tight, Direction::Forward), 0);
#endif
}

// miopenFindSolutions without a preallocated workspace allocates what the query reports, so it
// finds GemmFwdRest while its workspace fits the budget, and not once it does not.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindSolutionsAllocatesBudgetedWorkspace)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem        = ReproDilatedConv();
    const auto required = problem.SolverWorkspace(&get_handle(), Direction::Forward, "GemmFwdRest");
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    AvailableMemoryHandle roomy{2 * required};
    const auto found = problem.FindSolutions(&roomy);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].first, "GemmFwdRest");
    EXPECT_EQ(found[0].second, required);

    AvailableMemoryHandle tight{required / 2};
    EXPECT_TRUE(problem.FindSolutions(&tight).empty());
#endif
}

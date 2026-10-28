// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The workspace size stored in a find-db record must not reach the caller or the
// IsEnoughWorkspace() check. Shipped find-dbs have been found with wrong values, so
// GetSolutions() has to report what the solver actually requires: in immediate mode the
// workspace of the configuration Immediate runs, in Find the solver's maximum (GWSS).

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "../workspace.hpp"

#include <miopen/any_solver.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/db_getter.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_db.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>

#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct FindDbPathOverride
{
    explicit FindDbPathOverride(const miopen::fs::path& path)
        : cached_path(miopen::debug::testing_find_db_path_override()),
          cached_embed(miopen::debug::rordb_embed_fs_override())
    {
        miopen::debug::testing_find_db_path_override() = path;
        miopen::debug::rordb_embed_fs_override()       = true;
    }

    ~FindDbPathOverride()
    {
        miopen::debug::testing_find_db_path_override() = cached_path;
        miopen::debug::rordb_embed_fs_override()       = cached_embed;
    }

    FindDbPathOverride(const FindDbPathOverride&)            = delete;
    FindDbPathOverride& operator=(const FindDbPathOverride&) = delete;

private:
    std::optional<miopen::fs::path> cached_path;
    bool cached_embed;
};

struct RecordedSolver
{
    miopen::solver::Id id;
    std::size_t max_ws;       // GetWorkspaceSize(): what Find and GWSS use
    std::size_t immediate_ws; // workspace of the configuration Immediate runs
    std::size_t recorded_ws;  // deliberately wrong value written to the find-db
};

std::size_t MaxWorkspace(const miopen::ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const miopen::solver::Id& id)
{
    const auto& solver = id.GetSolver();
    return solver.MayNeedWorkspace() ? solver.GetWorkspaceSize(ctx, problem) : 0;
}

/// Resolves the configuration the same way Immediate does (CompileSolution).
std::size_t ImmediateWorkspace(const miopen::ExecutionContext& ctx,
                               const miopen::conv::ProblemDescription& problem,
                               const miopen::solver::Id& id)
{
    const auto& solver = id.GetSolver();
    if(!solver.MayNeedWorkspace())
        return 0;
    if(!solver.IsTunable())
        return solver.GetWorkspaceSize(ctx, problem);
    auto immediate_ctx                   = ctx;
    immediate_ctx.disable_search_enforce = true;
    // MakeConvDbGetter() is not exported; this is the same lazy GetDb() wrapper.
    auto perf_db                               = std::optional<miopen::PerformanceDb>{};
    std::function<miopen::PerformanceDb&()> db = [&]() -> miopen::PerformanceDb& {
        if(!perf_db)
            perf_db.emplace(miopen::GetDb(immediate_ctx));
        return *perf_db;
    };
    const auto solution = solver.FindSolution(immediate_ctx, problem, db, {});
    return solution.Succeeded() ? solution.workspace_sz : solver.GetWorkspaceSize(ctx, problem);
}

class GPU_FindDbWorkspace_FP32 : public ::testing::TestWithParam<miopen::conv::Direction>
{
protected:
    void SetUp() override
    {
        conv = miopen::ConvolutionDescriptor{
            2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
        conv.findMode.Set(miopen::FindMode::Values::Hybrid);

        x_desc = miopen::TensorDescriptor{miopenFloat, {4, 32, 28, 28}};
        w_desc = miopen::TensorDescriptor{miopenFloat, {64, 32, 3, 3}};
        y_desc = conv.GetForwardOutputTensor(x_desc, w_desc, miopenFloat);

        const auto direction = GetParam();
        problem              = direction == miopen::conv::Direction::Forward
                                   ? miopen::conv::ProblemDescription{x_desc, w_desc, y_desc, conv, direction}
                                   : miopen::conv::ProblemDescription{y_desc, w_desc, x_desc, conv, direction};
        ctx                  = miopen::ExecutionContext{&get_handle()};
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
    }

    /// Writes one find-db record for all applicable dynamic solvers, with a deliberately wrong
    /// workspace size. Solvers that need workspace come first (fastest recorded time) and
    /// alternate between overstated and understated sizes, starting with overstated. Solvers that
    /// need no workspace are recorded with a non-zero size.
    void WriteFindDb(const miopen::fs::path& path)
    {
        std::vector<miopen::solver::Id> with_ws, without_ws;
        for(const auto& id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
        {
            const auto& solver = id.GetSolver();
            if(solver.IsEmpty() || !solver.IsDynamic() || !solver.IsApplicable(ctx, problem))
                continue;
            (MaxWorkspace(ctx, problem, id) > 0 ? with_ws : without_ws).push_back(id);
        }

        constexpr std::size_t bogus = std::size_t{1} << 20;
        for(std::size_t i = 0; i < with_ws.size(); ++i)
        {
            const auto& id   = with_ws[i];
            const auto max   = MaxWorkspace(ctx, problem, id);
            const auto immed = ImmediateWorkspace(ctx, problem, id);
            recorded.push_back({id, max, immed, i % 2 == 0 ? max + bogus : immed / 2});
        }
        for(const auto& id : without_ws)
            recorded.push_back({id, 0, 0, bogus});

        std::ostringstream line;
        problem.Serialize(line);
        line << '=';
        for(std::size_t i = 0; i < recorded.size(); ++i)
        {
            const auto& r    = recorded[i];
            const auto value = miopen::FindDbData{
                static_cast<float>(i + 1), r.recorded_ws, r.id.GetAlgo(problem.GetDirection())};
            line << (i == 0 ? "" : ";") << r.id.ToString() << ':' << value;
        }

        std::ofstream file{path};
        file << line.str() << std::endl;
    }

    const RecordedSolver* FindRecorded(uint64_t solution_id) const
    {
        for(const auto& r : recorded)
            if(r.id.Value() == solution_id)
                return &r;
        return nullptr;
    }

    miopen::ConvolutionDescriptor conv;
    miopen::TensorDescriptor x_desc, w_desc, y_desc;
    miopen::conv::ProblemDescription problem;
    miopen::ExecutionContext ctx;
    std::vector<RecordedSolver> recorded;
};

} // namespace

TEST_P(GPU_FindDbWorkspace_FP32, ImmediateReportsWorkspaceOfImmediateConfig)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};

    WriteFindDb(temp_file.Path());
    if(recorded.empty())
        GTEST_SKIP() << "No applicable dynamic solver";

    // No InvokeParams: the path miopenConvolution*GetSolution takes.
    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, recorded.size(), &fallback);

    ASSERT_FALSE(solutions.empty());
    EXPECT_TRUE(fallback == miopen::FallbackPath::None) << "The find-db record was not used";
    auto n_wrong_records = 0;
    for(const auto& s : solutions)
    {
        const auto* r = FindRecorded(s.solution_id);
        ASSERT_NE(r, nullptr) << "Unexpected solver "
                              << miopen::solver::Id{s.solution_id}.ToString();
        EXPECT_EQ(s.workspace_size, r->immediate_ws) << r->id.ToString();
        // A buffer sized by GWSS or GetSolutionWorkspaceSize must fit the Immediate config.
        EXPECT_LE(r->immediate_ws, r->max_ws) << r->id.ToString();
        if(r->recorded_ws != r->immediate_ws)
            ++n_wrong_records;
    }
    EXPECT_GT(n_wrong_records, 0) << "No returned solution had a wrong find-db workspace";
}

TEST_P(GPU_FindDbWorkspace_FP32, OverstatedFindDbWorkspaceDoesNotDropSolver)
{
    if(GetParam() != miopen::conv::Direction::Forward)
        GTEST_SKIP() << "Forward only";

    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};

    WriteFindDb(temp_file.Path());
    if(recorded.empty() || recorded.front().max_ws == 0)
        GTEST_SKIP() << "No applicable dynamic solver needs workspace";

    // Provide exactly what GWSS returns for the fastest recorded solver. Its find-db record
    // overstates that, which used to make IsEnoughWorkspace() drop it.
    const auto& best = recorded.front();
    Workspace wspace{best.max_ws};
    const auto invoke_ctx = miopen::AnyInvokeParams{miopen::conv::DataInvokeParams{
        miopen::ConvFwdTensors{x_desc, nullptr, w_desc, nullptr, y_desc, nullptr},
        wspace.ptr(),
        wspace.size(),
        false}};

    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, 1, &fallback, &invoke_ctx);

    ASSERT_EQ(solutions.size(), 1u);
    EXPECT_TRUE(fallback == miopen::FallbackPath::None) << "The find-db record was not used";
    EXPECT_EQ(solutions.front().solution_id, best.id.Value())
        << "Got " << miopen::solver::Id{solutions.front().solution_id}.ToString() << ", expected "
        << best.id.ToString();
    EXPECT_EQ(solutions.front().workspace_size, best.max_ws);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_FindDbWorkspace_FP32,
                         testing::Values(miopen::conv::Direction::Forward,
                                         miopen::conv::Direction::BackwardData,
                                         miopen::conv::Direction::BackwardWeights));

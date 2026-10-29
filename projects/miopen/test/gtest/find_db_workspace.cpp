// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The workspace size stored in a find-db record must not reach the caller or the
// IsEnoughWorkspace() check. Shipped find-dbs have been found with wrong values, so
// GetSolutions() uses the solver's own requirement: IsEnoughWorkspace() checks the workspace of
// the configuration that would run (for tunable solvers, the perf-db or default configuration),
// immediate mode reports that size, and Find reports the solver's maximum (GWSS).

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "find_db.hpp"

#include <miopen/any_solver.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/convolution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_db.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace miopen {
// Resolves each solution's configuration the way Immediate does (src/hip/convolutionocl.cpp).
std::vector<solver::ConvSolution>
GetConvSolutions(const ExecutionContext& ctx,
                 const conv::ProblemDescription& problem,
                 const std::vector<miopenConvSolution_t> solutions);
} // namespace miopen

namespace {

struct FindDbPathOverride
{
    explicit FindDbPathOverride(const miopen::fs::path& path)
        : cached(miopen::debug::testing_find_db_path_override())
    {
        miopen::debug::testing_find_db_path_override() = path;
    }

    ~FindDbPathOverride() { miopen::debug::testing_find_db_path_override() = cached; }

    FindDbPathOverride(const FindDbPathOverride&)            = delete;
    FindDbPathOverride& operator=(const FindDbPathOverride&) = delete;

private:
    std::optional<miopen::fs::path> cached;
    miopen::TestRordbEmbedFsOverrideLock rordb_embed_fs_override;
};

struct RecordedSolver
{
    miopen::solver::Id id;
    std::size_t max_ws;      // GetWorkspaceSize(): what Find reports and GWSS returns
    std::size_t run_ws;      // workspace of the configuration that would run
    std::size_t recorded_ws; // value written to the find-db
};

std::size_t MaxWorkspace(const miopen::ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const miopen::solver::Id& id)
{
    const auto& solver = id.GetSolver();
    return solver.MayNeedWorkspace() ? solver.GetWorkspaceSize(ctx, problem) : 0;
}

std::size_t RunWorkspace(const miopen::ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const miopen::solver::Id& id)
{
    const auto& solver = id.GetSolver();
    if(!solver.MayNeedWorkspace())
        return 0;
    if(!solver.IsTunable())
        return solver.GetWorkspaceSize(ctx, problem);
    auto run_ctx                   = ctx;
    run_ctx.do_search              = false;
    run_ctx.disable_search_enforce = true;
    const auto solutions           = miopen::GetConvSolutions(
        run_ctx, problem, {miopenConvSolution_t{0.0f, 0, id.Value(), id.GetAlgo()}});
    return solutions.front().Succeeded() ? solutions.front().workspace_sz
                                         : solver.GetWorkspaceSize(ctx, problem);
}

class FindDbWorkspaceTest
{
protected:
    void SetUpProblem(miopen::conv::Direction direction,
                      miopenDataType_t type,
                      miopenTensorLayout_t layout,
                      int group_count,
                      const std::vector<std::size_t>& x_lens,
                      const std::vector<std::size_t>& w_lens)
    {
        conv = miopen::ConvolutionDescriptor{
            2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
        conv.group_count = group_count;
        conv.findMode.Set(miopen::FindMode::Values::Hybrid);

        x_desc = miopen::TensorDescriptor{type, layout, x_lens};
        w_desc = miopen::TensorDescriptor{type, layout, w_lens};
        y_desc = conv.GetForwardOutputTensorWithLayout(
            x_desc, w_desc, layout == miopenTensorNHWC ? "NHWC" : "NCHW", type);

        problem = direction == miopen::conv::Direction::Forward
                      ? miopen::conv::ProblemDescription{x_desc, w_desc, y_desc, conv, direction}
                      : miopen::conv::ProblemDescription{y_desc, w_desc, x_desc, conv, direction};
        ctx     = miopen::ExecutionContext{&get_handle()};
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
    }

    std::vector<miopen::solver::Id> ApplicableDynamicSolvers() const
    {
        std::vector<miopen::solver::Id> ids;
        for(const auto& id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
        {
            const auto& solver = id.GetSolver();
            if(!solver.IsEmpty() && solver.IsDynamic() && solver.IsApplicable(ctx, problem))
                ids.push_back(id);
        }
        return ids;
    }

    RecordedSolver Record(const miopen::solver::Id& id, std::size_t recorded_ws) const
    {
        return {id, MaxWorkspace(ctx, problem, id), RunWorkspace(ctx, problem, id), recorded_ws};
    }

    /// Writes one find-db record with the solvers in `recorded` order (fastest first).
    void WriteFindDb(const miopen::fs::path& path) const
    {
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

    /// InvokeParams as Find passes them; only the workspace matters to GetSolutions().
    miopen::AnyInvokeParams InvokeParams(Workspace& wspace) const
    {
        switch(problem.GetDirection())
        {
        case miopen::conv::Direction::Forward:
            return miopen::AnyInvokeParams{miopen::conv::DataInvokeParams{
                miopen::ConvFwdTensors{x_desc, nullptr, w_desc, nullptr, y_desc, nullptr},
                wspace.ptr(),
                wspace.size(),
                false}};
        case miopen::conv::Direction::BackwardData:
            return miopen::AnyInvokeParams{miopen::conv::DataInvokeParams{
                miopen::ConvBwdTensors{y_desc, nullptr, w_desc, nullptr, x_desc, nullptr},
                wspace.ptr(),
                wspace.size(),
                false}};
        case miopen::conv::Direction::BackwardWeights:
            return miopen::AnyInvokeParams{miopen::conv::WrWInvokeParams{
                miopen::ConvWrwTensors{y_desc, nullptr, x_desc, nullptr, w_desc, nullptr},
                wspace.ptr(),
                wspace.size(),
                false}};
        }
        return {};
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

class GPU_FindDbWorkspace_FP32 : public FindDbWorkspaceTest,
                                 public ::testing::TestWithParam<miopen::conv::Direction>
{
protected:
    void SetUp() override
    {
        SetUpProblem(GetParam(), miopenFloat, miopenTensorNCHW, 1, {4, 32, 28, 28}, {64, 32, 3, 3});
    }

    /// Records every applicable dynamic solver. Solvers that need workspace come first (fastest
    /// recorded time); `wrong_ws` gives each one's stored size. Solvers that need no workspace
    /// are recorded with a non-zero size.
    template <class F>
    void RecordAll(F wrong_ws)
    {
        std::vector<miopen::solver::Id> with_ws, without_ws;
        for(const auto& id : ApplicableDynamicSolvers())
            (MaxWorkspace(ctx, problem, id) > 0 ? with_ws : without_ws).push_back(id);
        for(std::size_t i = 0; i < with_ws.size(); ++i)
        {
            auto r        = Record(with_ws[i], 0);
            r.recorded_ws = wrong_ws(i, r);
            recorded.push_back(r);
        }
        for(const auto& id : without_ws)
            recorded.push_back(Record(id, std::size_t{1} << 20));
    }
};

class GPU_FindDbWorkspaceCkWrw_BFP16 : public FindDbWorkspaceTest, public ::testing::Test
{
protected:
    void SetUp() override
    {
        // NHWC backward-weights where CK's workspace depends on the configuration (split-K). On
        // gfx950 the system perf-db's configuration for this problem needs no workspace while
        // the maximum over configurations does.
        SetUpProblem(miopen::conv::Direction::BackwardWeights,
                     miopenBFloat16,
                     miopenTensorNHWC,
                     1,
                     {8, 64, 75, 75},
                     {64, 64, 3, 3});
    }
};

} // namespace

TEST_P(GPU_FindDbWorkspace_FP32, ImmediateReportsRunWorkspace)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    constexpr std::size_t bogus = std::size_t{1} << 20;
    RecordAll([](std::size_t i, const RecordedSolver& r) {
        return i % 2 == 0 ? r.max_ws + bogus : r.run_ws / 2;
    });
    ASSERT_FALSE(recorded.empty()) << "No applicable dynamic solver";
    WriteFindDb(temp_file.Path());

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
        EXPECT_EQ(s.workspace_size, r->run_ws) << r->id.ToString();
        // A buffer sized by GWSS or GetSolutionWorkspaceSize must fit the configuration.
        EXPECT_LE(r->run_ws, r->max_ws) << r->id.ToString();
        if(r->recorded_ws != r->run_ws)
            ++n_wrong_records;
    }
    EXPECT_GT(n_wrong_records, 0) << "No returned solution had a wrong find-db workspace";
}

TEST_P(GPU_FindDbWorkspace_FP32, FindKeepsSolverWhenStoredWorkspaceIsTooLarge)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    RecordAll([](std::size_t, const RecordedSolver& r) { return r.max_ws + (1 << 20); });
    ASSERT_FALSE(recorded.empty()) << "No applicable dynamic solver";
    WriteFindDb(temp_file.Path());

    // Provide exactly what the fastest recorded solver's configuration needs. Its find-db
    // record overstates that, which used to make IsEnoughWorkspace() drop it.
    const auto& best = recorded.front();
    Workspace wspace{best.run_ws};
    const auto invoke_ctx = InvokeParams(wspace);

    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, 1, &fallback, &invoke_ctx);

    ASSERT_EQ(solutions.size(), 1u);
    EXPECT_TRUE(fallback == miopen::FallbackPath::None) << "The find-db record was not used";
    EXPECT_EQ(solutions.front().solution_id, best.id.Value())
        << "Got " << miopen::solver::Id{solutions.front().solution_id}.ToString() << ", expected "
        << best.id.ToString();
    // Find reports the maximum over configurations, like GWSS.
    EXPECT_EQ(solutions.front().workspace_size, best.max_ws);
}

TEST_P(GPU_FindDbWorkspace_FP32, FindDropsSolverWhenStoredWorkspaceIsTooSmall)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    RecordAll([](std::size_t, const RecordedSolver&) { return std::size_t{0}; });
    if(recorded.empty() || recorded.front().run_ws == 0)
        GTEST_SKIP() << "No applicable dynamic solver needs workspace";
    WriteFindDb(temp_file.Path());

    // The fastest recorded solver's record says it needs nothing, but its configuration does.
    // A buffer one byte short of that must not select it.
    const auto& best = recorded.front();
    Workspace wspace{best.run_ws - 1};
    const auto invoke_ctx = InvokeParams(wspace);

    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, 1, &fallback, &invoke_ctx);

    for(const auto& s : solutions)
    {
        EXPECT_NE(s.solution_id, best.id.Value()) << best.id.ToString() << " does not fit";
        const auto* r = FindRecorded(s.solution_id);
        ASSERT_NE(r, nullptr);
        EXPECT_LE(r->run_ws, wspace.size()) << r->id.ToString();
    }
}

TEST_F(GPU_FindDbWorkspaceCkWrw_BFP16, FindKeepsTunedSolverThatFits)
{
    const auto ck = miopen::solver::Id{"ConvHipImplicitGemmGroupWrwXdlops"};
    if(!ck.IsValid() || !ck.GetSolver().IsApplicable(ctx, problem))
        GTEST_SKIP() << "ConvHipImplicitGemmGroupWrwXdlops is not applicable here";
    const auto naive = miopen::solver::Id{"ConvDirectNaiveConvWrw"};

    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    recorded.push_back(Record(ck, 0));
    recorded.front().recorded_ws = recorded.front().run_ws; // a correct record
    if(naive.IsValid() && naive.GetSolver().IsApplicable(ctx, problem))
        recorded.push_back(Record(naive, 0));
    const auto& best = recorded.front();
    if(best.run_ws >= best.max_ws)
        GTEST_SKIP() << "The configuration that runs needs the maximum workspace (" << best.max_ws
                     << "); nothing to distinguish";
    WriteFindDb(temp_file.Path());

    // A buffer that fits the configuration that runs but not the maximum over configurations
    // must keep the CK solver.
    Workspace wspace{best.run_ws};
    const auto invoke_ctx = InvokeParams(wspace);

    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, 1, &fallback, &invoke_ctx);

    ASSERT_EQ(solutions.size(), 1u);
    EXPECT_EQ(solutions.front().solution_id, ck.Value())
        << "Got " << miopen::solver::Id{solutions.front().solution_id}.ToString();
    EXPECT_EQ(solutions.front().workspace_size, best.max_ws);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_FindDbWorkspace_FP32,
                         testing::Values(miopen::conv::Direction::Forward,
                                         miopen::conv::Direction::BackwardData,
                                         miopen::conv::Direction::BackwardWeights));

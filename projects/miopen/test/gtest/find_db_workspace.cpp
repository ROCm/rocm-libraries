// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The workspace size stored in a find-db record must not reach the caller or the
// IsEnoughWorkspace() check. Shipped find-dbs have been found with wrong values, so
// GetSolutions() uses the solver's own requirement: IsEnoughWorkspace() checks the workspace of
// the configuration that would run (for a tunable solver whose workspace depends on the
// configuration, the perf-db or default configuration; for every other solver,
// GetWorkspaceSize()), immediate mode reports that size, and Find reports the solver's maximum
// (GWSS).

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "find_db.hpp"

#include <miopen/any_solver.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
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
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CK_PROBLEM_CACHE)

namespace miopen {
// Resolves each solution's configuration the way Immediate does (src/hip/convolutionocl.cpp).
std::vector<solver::ConvSolution>
GetConvSolutions(const ExecutionContext& ctx,
                 const conv::ProblemDescription& problem,
                 const std::vector<miopenConvSolution_t> solutions);
} // namespace miopen

namespace {

using miopen::FindDbPathOverride;

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

/// The layout name GetForwardOutputTensorWithLayout() takes; the tests use only these four.
const char* LayoutName(miopenTensorLayout_t layout)
{
    if(layout == miopenTensorNHWC)
        return "NHWC";
    if(layout == miopenTensorNDHWC)
        return "NDHWC";
    if(layout == miopenTensorNCDHW)
        return "NCDHW";
    return "NCHW";
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
        // 2D or 3D, from the lengths; padding, strides and dilations are 1 in every dimension.
        const auto spatial = x_lens.size() - 2;
        const std::vector<int> ones(spatial, 1);
        conv             = miopen::ConvolutionDescriptor{spatial,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             ones,
                                             ones,
                                             ones,
                                             std::vector<int>(spatial, 0)};
        conv.group_count = group_count;
        conv.findMode.Set(miopen::FindMode::Values::Hybrid);

        x_desc = miopen::TensorDescriptor{type, layout, x_lens};
        w_desc = miopen::TensorDescriptor{type, layout, w_lens};
        y_desc = conv.GetForwardOutputTensorWithLayout(x_desc, w_desc, LayoutName(layout), type);

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

/// A grouped convolution's shape; the lengths are in NCHW or NCDHW order, whatever the layout.
struct ConvShape
{
    int group_count;
    std::vector<std::size_t> x_lens;
    std::vector<std::size_t> w_lens;
};

std::ostream& operator<<(std::ostream& os, const ConvShape& p)
{
    os << 'g' << p.group_count;
    const auto lens = [&](const char* name, const std::vector<std::size_t>& v) {
        os << name;
        for(const auto n : v)
            os << '_' << n;
    };
    lens(" x", p.x_lens);
    lens(" w", p.w_lens);
    return os;
}

/// NHWC bf16 backward-weights problems for the CK grouped solver, whose workspace depends on the
/// configuration (split-K). A case is only meaningful where the configuration that runs (from the
/// system perf-db) needs less than the maximum over configurations; it skips elsewhere. The list
/// covers gfx942 and gfx950.
class GPU_FindDbWorkspaceCkWrw_BFP16 : public FindDbWorkspaceTest,
                                       public ::testing::TestWithParam<ConvShape>
{
protected:
    void SetUp() override
    {
        const auto& p = GetParam();
        SetUpProblem(miopen::conv::Direction::BackwardWeights,
                     miopenBFloat16,
                     miopenTensorNHWC,
                     p.group_count,
                     p.x_lens,
                     p.w_lens);
    }

    /// Records the CK solver first with a correct workspace, then the naive solver. Returns false
    /// (and skips) when the case cannot tell the configuration that runs from the maximum.
    bool RecordCkFirst()
    {
        const auto ck = miopen::solver::Id{"ConvHipImplicitGemmGroupWrwXdlops"};
        if(!ck.IsValid() || !ck.GetSolver().IsApplicable(ctx, problem))
        {
            skip_reason = "ConvHipImplicitGemmGroupWrwXdlops is not applicable here";
            return false;
        }
        recorded.push_back(Record(ck, 0));
        recorded.front().recorded_ws = recorded.front().run_ws; // a correct record
        const auto naive             = miopen::solver::Id{"ConvDirectNaiveConvWrw"};
        if(naive.IsValid() && naive.GetSolver().IsApplicable(ctx, problem))
            recorded.push_back(Record(naive, 0));
        const auto& best = recorded.front();
        if(best.run_ws >= best.max_ws)
        {
            std::ostringstream reason;
            reason << "The configuration that runs needs the maximum workspace (" << best.max_ws
                   << "); nothing to distinguish";
            skip_reason = reason.str();
            return false;
        }
        return true;
    }

    std::string skip_reason;
};

struct FlagProblem
{
    miopen::conv::Direction direction;
    miopenTensorLayout_t layout;
    ConvShape shape;
};

std::ostream& operator<<(std::ostream& os, const FlagProblem& p)
{
    return os << static_cast<int>(p.direction) << ' ' << LayoutName(p.layout) << ' ' << p.shape;
}

/// GetSolutions() resolves the configuration only for solvers whose WorkspaceDependsOnConfig()
/// is true and uses GetWorkspaceSize() for every other one. That is correct only if every other
/// tunable solver's configuration needs exactly GetWorkspaceSize(): a solver that needs more
/// would be run with too little workspace, one that needs less would be over-reported, and
/// Find would drop it for a buffer that fits. Every configuration must fit GetWorkspaceSize().
class FindDbWorkspaceFlagTest : public FindDbWorkspaceTest,
                                public ::testing::TestWithParam<FlagProblem>
{
protected:
    void Check(miopenDataType_t type)
    {
        const auto& p = GetParam();
        SetUpProblem(
            p.direction, type, p.layout, p.shape.group_count, p.shape.x_lens, p.shape.w_lens);
        auto n_checked = 0;
        for(const auto& id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
        {
            const auto& solver = id.GetSolver();
            if(solver.IsEmpty() || !solver.IsTunable() || !solver.MayNeedWorkspace() ||
               !solver.IsApplicable(ctx, problem))
                continue;
            ++n_checked;
            const auto max_ws = MaxWorkspace(ctx, problem, id);
            const auto run_ws = RunWorkspace(ctx, problem, id);
            if(solver.WorkspaceDependsOnConfig())
                EXPECT_LE(run_ws, max_ws) << id.ToString();
            else
                EXPECT_EQ(run_ws, max_ws)
                    << id.ToString()
                    << ": its configuration's workspace is not GetWorkspaceSize(); "
                       "override WorkspaceDependsOnConfig() to return true";
        }
        if(n_checked == 0)
            GTEST_SKIP() << "No applicable tunable solver needs workspace";
    }
};

class GPU_FindDbWorkspaceFlag_FP32 : public FindDbWorkspaceFlagTest
{
};
class GPU_FindDbWorkspaceFlag_FP16 : public FindDbWorkspaceFlagTest
{
};
class GPU_FindDbWorkspaceFlag_BFP16 : public FindDbWorkspaceFlagTest
{
};

std::vector<FlagProblem> FlagProblems()
{
    std::vector<FlagProblem> problems;
    for(const auto direction : {miopen::conv::Direction::Forward,
                                miopen::conv::Direction::BackwardData,
                                miopen::conv::Direction::BackwardWeights})
    {
        problems.push_back({direction, miopenTensorNCHW, {1, {2, 32, 28, 28}, {64, 32, 3, 3}}});
        problems.push_back({direction, miopenTensorNCHW, {1, {2, 64, 14, 14}, {128, 64, 1, 1}}});
        problems.push_back({direction, miopenTensorNHWC, {1, {2, 32, 28, 28}, {64, 32, 3, 3}}});
        problems.push_back({direction, miopenTensorNHWC, {4, {2, 64, 28, 28}, {64, 16, 3, 3}}});
        // 3D, so the flags of the 3D solvers (the CK grouped backward-weights one sets it) are
        // checked too; FP16 and BF16 reach the CK 3D solvers on gfx11 as well as on CDNA.
        problems.push_back(
            {direction, miopenTensorNDHWC, {1, {2, 32, 8, 14, 14}, {64, 32, 3, 3, 3}}});
        problems.push_back(
            {direction, miopenTensorNDHWC, {4, {2, 64, 8, 14, 14}, {64, 16, 3, 3, 3}}});
        problems.push_back(
            {direction, miopenTensorNCDHW, {1, {2, 32, 8, 14, 14}, {64, 32, 3, 3, 3}}});
        // ConvHipConv's smoke shape (4 channels per group), in both layouts it serves, so its flag
        // is checked where it applies (gfx950, gfx1250; FP16 and BF16).
        problems.push_back({direction, miopenTensorNHWC, {16, {4, 64, 8, 1}, {64, 4, 3, 3}}});
        problems.push_back({direction, miopenTensorNCHW, {16, {4, 64, 8, 1}, {64, 4, 3, 3}}});
    }
    return problems;
}

/// The CK backward-weights solvers take their untuned configuration (AI heuristics or default)
/// from DefaultConfigCache, and their maximum workspace from a per-problem memo, on every query
/// after the first for a problem. What a cache hit returns must be what is computed without the
/// caches, and the configuration must run with the same workspace.
class FindDbWorkspaceDefaultCacheTest : public FindDbWorkspaceTest,
                                        public ::testing::TestWithParam<ConvShape>
{
protected:
    void Check(miopenDataType_t type)
    {
        const auto& p   = GetParam();
        const auto is3d = p.x_lens.size() == 5;
        SetUpProblem(miopen::conv::Direction::BackwardWeights,
                     type,
                     is3d ? miopenTensorNDHWC : miopenTensorNHWC,
                     p.group_count,
                     p.x_lens,
                     p.w_lens);
        if(is3d)
            CheckSolver(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
        else
            CheckSolver(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
    }

private:
    template <class Solver>
    void CheckSolver(const Solver& solver)
    {
        if(!solver.IsApplicable(ctx, problem))
            GTEST_SKIP() << solver.SolverDbId() << " is not applicable";

        std::optional<decltype(solver.GetDefaultPerformanceConfig(ctx, problem))> uncached;
        std::size_t uncached_max_ws = 0;
        {
            ScopedEnvironment<bool> no_cache(MIOPEN_DEBUG_CK_PROBLEM_CACHE, false);
            uncached        = solver.GetDefaultPerformanceConfig(ctx, problem);
            uncached_max_ws = solver.GetWorkspaceSize(ctx, problem);
        }
        // Two queries with the cache on: the second is a hit whatever earlier tests cached.
        std::ignore       = solver.GetDefaultPerformanceConfig(ctx, problem);
        const auto hit    = solver.GetDefaultPerformanceConfig(ctx, problem);
        std::ignore       = solver.GetWorkspaceSize(ctx, problem);
        const auto max_ws = solver.GetWorkspaceSize(ctx, problem);
        const auto name   = solver.SolverDbId();

        ASSERT_FALSE(uncached->kernel_id.empty()) << name;
        EXPECT_EQ(hit.kernel_id, uncached->kernel_id) << name;
        EXPECT_EQ(hit.split_k, uncached->split_k) << name;
        EXPECT_EQ(hit.UseTF32(), uncached->UseTF32()) << name;
        EXPECT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, hit)) << name;
        EXPECT_EQ(max_ws, uncached_max_ws) << name;

        const auto from_hit      = solver.GetSolution(ctx, problem, hit);
        const auto from_uncached = solver.GetSolution(ctx, problem, *uncached);
        ASSERT_TRUE(from_uncached.Succeeded()) << name;
        ASSERT_TRUE(from_hit.Succeeded()) << name;
        EXPECT_EQ(from_hit.workspace_sz, from_uncached.workspace_sz) << name;
    }
};

class GPU_FindDbWorkspaceDefaultCache_FP32 : public FindDbWorkspaceDefaultCacheTest
{
};
class GPU_FindDbWorkspaceDefaultCache_FP16 : public FindDbWorkspaceDefaultCacheTest
{
};

std::vector<ConvShape> DefaultCacheProblems()
{
    return {{1, {2, 64, 28, 28}, {64, 64, 3, 3}},
            {4, {2, 64, 28, 28}, {64, 16, 3, 3}},
            {4, {2, 64, 8, 14, 14}, {64, 16, 3, 3, 3}}};
}

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

TEST_P(GPU_FindDbWorkspaceFlag_FP32, ConfigIndependentWorkspaceIsGetWorkspaceSize)
{
    Check(miopenFloat);
}

TEST_P(GPU_FindDbWorkspaceFlag_FP16, ConfigIndependentWorkspaceIsGetWorkspaceSize)
{
    Check(miopenHalf);
}

TEST_P(GPU_FindDbWorkspaceFlag_BFP16, ConfigIndependentWorkspaceIsGetWorkspaceSize)
{
    Check(miopenBFloat16);
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

TEST_P(GPU_FindDbWorkspaceCkWrw_BFP16, FindKeepsTunedSolverThatFits)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    if(!RecordCkFirst())
        GTEST_SKIP() << skip_reason;
    WriteFindDb(temp_file.Path());

    // A buffer that fits the configuration that runs but not the maximum over configurations
    // must keep the CK solver.
    const auto& best = recorded.front();
    Workspace wspace{best.run_ws};
    const auto invoke_ctx = InvokeParams(wspace);

    auto fallback        = miopen::FallbackPath::WTI;
    const auto solutions = conv.GetSolutions(ctx, problem, 1, &fallback, &invoke_ctx);

    ASSERT_EQ(solutions.size(), 1u);
    EXPECT_EQ(solutions.front().solution_id, best.id.Value())
        << "Got " << miopen::solver::Id{solutions.front().solution_id}.ToString();
    EXPECT_EQ(solutions.front().workspace_size, best.max_ws);
}

TEST_P(GPU_FindDbWorkspaceCkWrw_BFP16, WorkSpaceSizeReportsMaximumOnFindDbHit)
{
    const miopen::TempFile temp_file{"miopen.test.find_db_workspace"};
    const FindDbPathOverride path_override{temp_file.Path()};
    if(!RecordCkFirst())
        GTEST_SKIP() << skip_reason;
    WriteFindDb(temp_file.Path());

    // GWSS (Hybrid mode) takes the find-db's best solver and must return its maximum workspace,
    // so that any configuration fits, not the workspace of the configuration in the perf-db.
    EXPECT_EQ(conv.GetWorkSpaceSize(ctx, problem), recorded.front().max_ws);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_FindDbWorkspaceCkWrw_BFP16,
                         testing::Values(ConvShape{1, {2, 64, 56, 56}, {64, 64, 3, 3}},
                                         ConvShape{1, {2, 128, 28, 28}, {128, 128, 3, 3}}));

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_FindDbWorkspace_FP32,
                         testing::Values(miopen::conv::Direction::Forward,
                                         miopen::conv::Direction::BackwardData,
                                         miopen::conv::Direction::BackwardWeights));

INSTANTIATE_TEST_SUITE_P(Quick, GPU_FindDbWorkspaceFlag_FP32, testing::ValuesIn(FlagProblems()));
INSTANTIATE_TEST_SUITE_P(Quick, GPU_FindDbWorkspaceFlag_FP16, testing::ValuesIn(FlagProblems()));
INSTANTIATE_TEST_SUITE_P(Quick, GPU_FindDbWorkspaceFlag_BFP16, testing::ValuesIn(FlagProblems()));

TEST_P(GPU_FindDbWorkspaceDefaultCache_FP32, CacheHitIsUncachedDefault) { Check(miopenFloat); }

TEST_P(GPU_FindDbWorkspaceDefaultCache_FP16, CacheHitIsUncachedDefault) { Check(miopenHalf); }

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_FindDbWorkspaceDefaultCache_FP32,
                         testing::ValuesIn(DefaultCacheProblems()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_FindDbWorkspaceDefaultCache_FP16,
                         testing::ValuesIn(DefaultCacheProblems()));

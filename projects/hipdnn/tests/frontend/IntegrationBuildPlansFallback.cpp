// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Engine fallback in Graph::build_plans(HEURISTICS_CHOICE) when the top-ranked engine
// accepts a graph (it is returned by GetApplicableEngineIds) but cannot build an execution
// plan for it.
//
// Any failure to finalize the active plan falls back to another ranked engine (scanning from the top) and is
// logged at WARN. Here the failure is either a plain plan-creation failure ("no solver")
// or an out-of-memory status, and the failing engine is the user's soft preferred engine.
// When no plan finalizes, build_plans() returns GRAPH_EXECUTION_PLAN_CREATION_FAILED with
// every attempted engine's failure. BuildPlanPolicy::ALL skips every failing plan.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

const std::vector<int64_t> K_DIMS = {1, 3, 4, 4};

// Shared scenario for a failing engine that is the soft preferred engine. Concrete suites
// choose which fake plugin fails and whether a healthy engine is also loaded.
class IntegrationBuildPlansFallbackBase : public hipdnn_tests::IntegrationTestFixture
{
protected:
    // The engine made the active plan through the soft preferred-engine setter, so the
    // tests do not depend on heuristic order.
    virtual int64_t failingEngineId() const = 0;

    // The plugin's own failure text, which the reported error must carry.
    virtual const char* failureText() const = 0;

    static int64_t noSolverEngineId()
    {
        return hipdnn_tests::plugin_constants::engineId<NoSolverPlugin>();
    }

    static int64_t resourceFailEngineId()
    {
        return hipdnn_tests::plugin_constants::engineId<ResourceFailPlugin>();
    }

    static int64_t goodEngineId()
    {
        return hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    }

    std::shared_ptr<Graph> createBuiltGraph(const std::string& name)
    {
        auto graph = std::make_shared<Graph>();
        graph->set_name(name)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1)
            .set_name("X")
            .set_dim(K_DIMS)
            .set_stride({K_DIMS[1] * K_DIMS[2] * K_DIMS[3], K_DIMS[2] * K_DIMS[3], K_DIMS[3], 1})
            .set_data_type(DataType::FLOAT);

        PointwiseAttributes attrs;
        attrs.set_name("relu_node");
        attrs.set_mode(PointwiseMode::RELU_FWD);

        auto y = graph->pointwise(x, attrs);
        y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

        graph->set_preferred_engine_id_ext(failingEngineId());

        auto result = graph->validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        return graph;
    }

    int64_t activeEngineId(Graph& graph)
    {
        std::string planName;
        const auto result = graph.get_plan_name(_handle, planName);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        return engineNameOrIdToId(planName);
    }

    // The plan that served the graph must be finalized and usable, not merely
    // named: its engine is reported, it answers a workspace query, and it executes.
    void expectPlanIsFinalizedOnEngine(Graph& graph, int64_t expectedEngineId)
    {
        EXPECT_EQ(activeEngineId(graph), expectedEngineId);

        int64_t selectedEngineId = 0;
        auto result = graph.get_execution_plan_engine_id(selectedEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(selectedEngineId, expectedEngineId);

        int64_t workspaceSize = -1;
        result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_GE(workspaceSize, 0);

        Tensor<float> xTensor(K_DIMS);
        Tensor<float> yTensor(K_DIMS);
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[1] = xTensor.memory().deviceData();
        variantPack[2] = yTensor.memory().deviceData();

        result = graph.execute(_handle, variantPack, nullptr);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    // Both engines are offered; the preferred one fails to finalize. build_plans() must
    // move on to the engine that works.
    void expectBuildPlansFallsBack(const std::string& name)
    {
        auto graph = createBuiltGraph(name);

        std::vector<int64_t> ranked;
        auto result = graph->get_ranked_engine_ids(ranked);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(ranked.size(), 2u) << "both engines must be offered for this scenario";

        result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
    }

    // Same scenario through the one-call build().
    void expectOneCallBuildFallsBack(const std::string& name)
    {
        auto graph = createBuiltGraph(name);

        auto result = graph->build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
    }

    // Control: BuildPlanPolicy::ALL skips a plan that fails to finalize.
    void expectPolicyAllSkipsFailingEngine(const std::string& name)
    {
        auto graph = createBuiltGraph(name);

        auto result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans(BuildPlanPolicy::ALL);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
    }

    // With no healthy engine to fall back to, build_plans() reports that no plan finalized
    // together with the failing engine's own error, and leaves the active plan unchanged.
    void expectBuildPlansReportsAggregatedError(const std::string& name)
    {
        auto graph = createBuiltGraph(name);

        auto result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        EXPECT_NE(result.err_msg.find("Failed to finalize any execution plan"), std::string::npos)
            << result.err_msg;
        EXPECT_NE(result.err_msg.find(failureText()), std::string::npos) << result.err_msg;

        EXPECT_EQ(activeEngineId(*graph), failingEngineId());
    }

    // Same scenario through the one-call build().
    void expectOneCallBuildReportsAggregatedError(const std::string& name)
    {
        auto graph = createBuiltGraph(name);

        auto result = graph->build(_handle);
        EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        EXPECT_NE(result.err_msg.find("Failed to finalize any execution plan"), std::string::npos)
            << result.err_msg;
        EXPECT_NE(result.err_msg.find(failureText()), std::string::npos) << result.err_msg;
    }
};

// The preferred engine fails plan creation with a plain failure; a healthy engine is loaded.
class IntegrationBuildPlansNoSolver : public IntegrationBuildPlansFallbackBase
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoSolverPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    int64_t failingEngineId() const override
    {
        return noSolverEngineId();
    }

    const char* failureText() const override
    {
        return "no applicable solver";
    }
};

TEST_F(IntegrationBuildPlansNoSolver, BuildFallsBackFromEngineWithNoSolver)
{
    expectBuildPlansFallsBack("BuildFallsBackFromEngineWithNoSolver");
}

TEST_F(IntegrationBuildPlansNoSolver, OneCallBuildFallsBackFromEngineWithNoSolver)
{
    expectOneCallBuildFallsBack("OneCallBuildFallsBackFromEngineWithNoSolver");
}

TEST_F(IntegrationBuildPlansNoSolver, PolicyAllSkipsEngineWithNoSolver)
{
    expectPolicyAllSkipsFailingEngine("PolicyAllSkipsEngineWithNoSolver");
}

// The preferred engine fails plan creation with an out-of-memory status; a healthy engine
// is loaded. Any failure falls back, whatever its kind.
class IntegrationBuildPlansResourceFailure : public IntegrationBuildPlansFallbackBase
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testResourceFailPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    int64_t failingEngineId() const override
    {
        return resourceFailEngineId();
    }

    const char* failureText() const override
    {
        return "out of memory";
    }
};

TEST_F(IntegrationBuildPlansResourceFailure, BuildFallsBackFromEngineWithResourceFailure)
{
    expectBuildPlansFallsBack("BuildFallsBackFromEngineWithResourceFailure");
}

TEST_F(IntegrationBuildPlansResourceFailure, OneCallBuildFallsBackFromEngineWithResourceFailure)
{
    expectOneCallBuildFallsBack("OneCallBuildFallsBackFromEngineWithResourceFailure");
}

TEST_F(IntegrationBuildPlansResourceFailure, PolicyAllSkipsEngineWithResourceFailure)
{
    expectPolicyAllSkipsFailingEngine("PolicyAllSkipsEngineWithResourceFailure");
}

// Only the plain-failure engine is loaded: nothing to fall back to.
class IntegrationBuildPlansNoSolverOnly : public IntegrationBuildPlansNoSolver
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoSolverPluginPath()};
    }
};

TEST_F(IntegrationBuildPlansNoSolverOnly, BuildPlansReportsErrorWhenNoEngineFinalizes)
{
    expectBuildPlansReportsAggregatedError("BuildPlansReportsErrorWhenNoEngineFinalizes");
}

TEST_F(IntegrationBuildPlansNoSolverOnly, OneCallBuildReportsErrorWhenNoEngineFinalizes)
{
    expectOneCallBuildReportsAggregatedError("OneCallBuildReportsErrorWhenNoEngineFinalizes");
}

// Only the out-of-memory engine is loaded: nothing to fall back to.
class IntegrationBuildPlansResourceFailureOnly : public IntegrationBuildPlansResourceFailure
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testResourceFailPluginPath()};
    }
};

TEST_F(IntegrationBuildPlansResourceFailureOnly, BuildPlansReportsErrorWhenNoEngineFinalizes)
{
    expectBuildPlansReportsAggregatedError("BuildPlansReportsErrorWhenNoEngineFinalizes");
}

TEST_F(IntegrationBuildPlansResourceFailureOnly, OneCallBuildReportsErrorWhenNoEngineFinalizes)
{
    expectOneCallBuildReportsAggregatedError("OneCallBuildReportsErrorWhenNoEngineFinalizes");
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Engine fallback in Graph::build_plans(HEURISTICS_CHOICE) when the top-ranked engine
// accepts a graph (it is returned by GetApplicableEngineIds) but cannot build an execution
// plan for it.
//
// Only the "no solver" outcome falls back to the next ranked engine: the plugin answers
// plan creation with HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE. Any other plan-creation failure
// (here an out-of-memory status) stops and is returned, even when the failing engine is
// the user's soft preferred engine. BuildPlanPolicy::ALL skips every failing plan.

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

class IntegrationBuildPlansFallback : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoSolverPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    // The engine made the active plan through the soft preferred-engine setter, so the
    // tests do not depend on heuristic order.
    virtual int64_t preferredEngineId() const
    {
        return noSolverEngineId();
    }

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

        graph->set_preferred_engine_id_ext(preferredEngineId());

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
};

// Both engines are offered; the preferred one has no solver. Plan building must move
// on to the engine that does.
TEST_F(IntegrationBuildPlansFallback, BuildFallsThroughEngineWithNoSolver)
{
    auto graph = createBuiltGraph("BuildFallsThroughEngineWithNoSolver");

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
TEST_F(IntegrationBuildPlansFallback, OneCallBuildFallsThroughEngineWithNoSolver)
{
    auto graph = createBuiltGraph("OneCallBuildFallsThroughEngineWithNoSolver");

    auto result = graph->build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
}

// Control: BuildPlanPolicy::ALL skips a plan that fails to finalize.
TEST_F(IntegrationBuildPlansFallback, PolicyAllSkipsEngineWithNoSolver)
{
    auto graph = createBuiltGraph("PolicyAllSkipsEngineWithNoSolver");

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
}

// With no healthy engine to fall back to, the failing engine's error is reported.
class IntegrationBuildPlansNoSolverOnly : public IntegrationBuildPlansFallback
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoSolverPluginPath()};
    }
};

TEST_F(IntegrationBuildPlansNoSolverOnly, BuildReportsErrorWhenNoEngineHasASolver)
{
    auto graph = createBuiltGraph("BuildReportsErrorWhenNoEngineHasASolver");

    auto result = graph->build(_handle);
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.err_msg.find("no applicable solver"), std::string::npos) << result.err_msg;
}

// The preferred engine fails with something other than "no solver" (an out-of-memory
// status from plan creation). That is a fault, not a decline: plan building stops with
// the engine's error and does not fall back to the healthy engine.
class IntegrationBuildPlansResourceFailure : public IntegrationBuildPlansFallback
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testResourceFailPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    int64_t preferredEngineId() const override
    {
        return resourceFailEngineId();
    }
};

TEST_F(IntegrationBuildPlansResourceFailure, BuildStopsAtEngineWithResourceFailure)
{
    auto graph = createBuiltGraph("BuildStopsAtEngineWithResourceFailure");

    std::vector<int64_t> ranked;
    auto result = graph->get_ranked_engine_ids(ranked);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(ranked.size(), 2u) << "both engines must be offered for this scenario";

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.err_msg.find("out of memory"), std::string::npos) << result.err_msg;

    // No fallback happened: the failing engine is still the active one.
    int64_t selectedEngineId = 0;
    result = graph->get_execution_plan_engine_id(selectedEngineId);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(selectedEngineId, resourceFailEngineId());
    EXPECT_EQ(activeEngineId(*graph), resourceFailEngineId());
}

// Same scenario through the one-call build().
TEST_F(IntegrationBuildPlansResourceFailure, OneCallBuildStopsAtEngineWithResourceFailure)
{
    auto graph = createBuiltGraph("OneCallBuildStopsAtEngineWithResourceFailure");

    auto result = graph->build(_handle);
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.err_msg.find("out of memory"), std::string::npos) << result.err_msg;
}

// Control: BuildPlanPolicy::ALL skips every failing plan, whatever the failure.
TEST_F(IntegrationBuildPlansResourceFailure, PolicyAllSkipsEngineWithResourceFailure)
{
    auto graph = createBuiltGraph("PolicyAllSkipsEngineWithResourceFailure");

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    expectPlanIsFinalizedOnEngine(*graph, goodEngineId());
}

} // namespace

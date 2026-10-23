// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Engine fallback when the top-ranked engine accepts a graph (it is returned by
// GetApplicableEngineIds) but cannot build an execution plan for it ("no solver").
// A second, healthy engine is loaded alongside; the graph must still build and
// run on it. ROCm/rocm-libraries#14266.

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

class IntegrationBuildPlansFallback : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoSolverPluginPath(),
                hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    static int64_t noSolverEngineId()
    {
        return hipdnn_tests::plugin_constants::engineId<NoSolverPlugin>();
    }

    static int64_t goodEngineId()
    {
        return hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    }

    // The no-solver engine is pinned to the front of the ranked list through the
    // soft preferred-engine setter, so the test does not depend on heuristic order.
    std::shared_ptr<Graph> createBuiltGraph(const std::string& name)
    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};

        auto graph = std::make_shared<Graph>();
        graph->set_name(name)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1)
            .set_name("X")
            .set_dim(dims)
            .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
            .set_data_type(DataType::FLOAT);

        PointwiseAttributes attrs;
        attrs.set_name("relu_node");
        attrs.set_mode(PointwiseMode::RELU_FWD);

        auto y = graph->pointwise(x, attrs);
        y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

        graph->set_preferred_engine_id_ext(noSolverEngineId());

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
};

// Both engines are offered; the pinned one has no solver. Plan building must move
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
    EXPECT_EQ(activeEngineId(*graph), goodEngineId());
}

// Same scenario through the one-call build().
TEST_F(IntegrationBuildPlansFallback, OneCallBuildFallsThroughEngineWithNoSolver)
{
    auto graph = createBuiltGraph("OneCallBuildFallsThroughEngineWithNoSolver");

    auto result = graph->build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(activeEngineId(*graph), goodEngineId());
}

// Control: BuildPlanPolicy::ALL already skips a plan that fails to finalize.
TEST_F(IntegrationBuildPlansFallback, PolicyAllSkipsEngineWithNoSolver)
{
    auto graph = createBuiltGraph("PolicyAllSkipsEngineWithNoSolver");

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(activeEngineId(*graph), goodEngineId());
}

} // namespace

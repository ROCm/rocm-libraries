// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v0.x builders' engine count, dense engine index, heuristic rank and
// per-engine plan failures against the real hipDNN backend and the test engine
// plugins. The test heuristic policy reverses hipDNN's candidate order so rank
// and index differ. Engine queries need hipDNN heuristics, which query the
// device, so these tests skip on device-less hosts.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8BuilderGraph.hpp"

namespace
{
namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace plugin_constants = hipdnn_tests::plugin_constants;
using hipdnn_shim_v8_test::DeviceBuffer;
using hipdnn_shim_v8_test::jsonEngineId;
using hipdnn_shim_v8_test::lastError;
using hipdnn_shim_v8_test::statusOf;

using hipdnn_shim_v8_test::floatTensorBytes;
using hipdnn_shim_v8_test::RELU_X_UID;
using hipdnn_shim_v8_test::RELU_Y_UID;
using hipdnn_shim_v8_test::X_DIMS;

class IntegrationCudnnShimV8BuilderEngines : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::int64_t engineCount()
    {
        const v8fe::OperationGraph_v8 graph
            = hipdnn_shim_v8_test::reluGraph(_handle, RELU_X_UID, RELU_Y_UID, X_DIMS);
        return graph.getEngineCount();
    }
};

TEST_F(IntegrationCudnnShimV8BuilderEngines, GoodPluginHasOneEngine)
{
    EXPECT_EQ(engineCount(), 1);
}

TEST_F(IntegrationCudnnShimV8BuilderEngines, RepeatedQueriesAgree)
{
    const v8fe::OperationGraph_v8 graph
        = hipdnn_shim_v8_test::reluGraph(_handle, RELU_X_UID, RELU_Y_UID, X_DIMS);
    EXPECT_EQ(graph.getEngineCount(), 1);
    EXPECT_EQ(graph.getEngineCount(), 1);
}

class IntegrationCudnnShimV8BuilderEnginesThreePlugins : public IntegrationCudnnShimV8BuilderEngines
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testHashedNamePluginPath()};
    }
};

TEST_F(IntegrationCudnnShimV8BuilderEnginesThreePlugins, EveryLoadedEngineCounted)
{
    EXPECT_EQ(engineCount(), 3);
}

class IntegrationCudnnShimV8BuilderEnginesNoEngines : public IntegrationCudnnShimV8BuilderEngines
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testNoApplicableEnginesAPluginPath()};
    }
};

TEST_F(IntegrationCudnnShimV8BuilderEnginesNoEngines, ZeroIsNotAnError)
{
    EXPECT_EQ(engineCount(), 0);
}

// Engines sorted by signed id: -32, -31, -2, then the execute-fails engine's positive id.
constexpr std::int64_t K_ENGINE_COUNT = 4;
constexpr std::int64_t K_CONTEXT_FAILS_INDEX = 0;
constexpr std::int64_t K_EXECUTE_FAILS_INDEX = 3;

class IntegrationCudnnShimV8BuilderEnginesMultiEngine : public hipdnn_shim_v8_test::ReluGraphFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testCudnnShimV8PluginPath()};
    }
};

TEST_F(IntegrationCudnnShimV8BuilderEnginesMultiEngine, IndexFollowsSignedEngineId)
{
    ASSERT_EQ(_graph->getEngineCount(), K_ENGINE_COUNT);

    const std::array<std::int64_t, 3> builtIds = {plugin_constants::engineId<CudnnShimV8Plugin>(),
                                                  plugin_constants::engineId<GoodPlugin>(),
                                                  plugin_constants::engineId<ExecuteFailsPlugin>()};
    for(std::int64_t index = K_CONTEXT_FAILS_INDEX + 1; index < K_ENGINE_COUNT; ++index)
    {
        SCOPED_TRACE(index);
        const v8fe::ExecutionPlan_v8 plan = hipdnn_shim_v8_test::planFor(*_graph, index);
        EXPECT_EQ(hipdnn_shim_v8_test::tagEngineIndex(plan.getTag()), index);
        EXPECT_EQ(jsonEngineId(plan.getJsonRepresentation()),
                  std::to_string(builtIds.at(static_cast<std::size_t>(index - 1))));
    }
}

// A plan's tag names its config's dense index; the one config whose plan cannot be
// built is the context-failing engine's.
TEST_F(IntegrationCudnnShimV8BuilderEnginesMultiEngine, ResultsFollowHeuristicRank)
{
    v8fe::EngineConfigList configs = hipdnn_shim_v8_test::heuristicConfigs(*_graph);
    ASSERT_EQ(static_cast<std::int64_t>(configs.size()), K_ENGINE_COUNT);

    std::vector<std::int64_t> rankedIndices;
    rankedIndices.reserve(configs.size());
    std::size_t failures = 0;
    for(v8fe::ManagedOpaqueDescriptor& config : configs)
    {
        const cudnnStatus_t status = statusOf([&] {
            const v8fe::ExecutionPlan_v8 plan
                = v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config, _graph->getTag()).build();
            rankedIndices.push_back(hipdnn_shim_v8_test::tagEngineIndex(plan.getTag()));
        });
        if(status != CUDNN_STATUS_SUCCESS)
        {
            ++failures;
            rankedIndices.push_back(K_CONTEXT_FAILS_INDEX);
        }
    }
    EXPECT_EQ(failures, 1U);

    std::vector<std::int64_t> indexOrder(configs.size());
    std::iota(indexOrder.begin(), indexOrder.end(), 0);
    EXPECT_TRUE(
        std::is_permutation(rankedIndices.begin(), rankedIndices.end(), indexOrder.begin()));
    EXPECT_NE(rankedIndices, indexOrder);
}

TEST_F(IntegrationCudnnShimV8BuilderEnginesMultiEngine, ContextFailureFailsOnlyItsPlan)
{
    for(std::int64_t index = 0; index < K_ENGINE_COUNT; ++index)
    {
        SCOPED_TRACE(index);
        const cudnnStatus_t status = statusOf(
            [&] { [[maybe_unused]] auto plan = hipdnn_shim_v8_test::planFor(*_graph, index); });
        if(index == K_CONTEXT_FAILS_INDEX)
        {
            EXPECT_NE(status, CUDNN_STATUS_SUCCESS);
        }
        else
        {
            EXPECT_EQ(status, CUDNN_STATUS_SUCCESS) << lastError();
        }
    }
}

TEST_F(IntegrationCudnnShimV8BuilderEnginesMultiEngine, ExecuteFailureReported)
{
    const v8fe::ExecutionPlan_v8 plan
        = hipdnn_shim_v8_test::planFor(*_graph, K_EXECUTE_FAILS_INDEX);

    const DeviceBuffer x(floatTensorBytes(X_DIMS));
    const DeviceBuffer y(floatTensorBytes(X_DIMS));
    const DeviceBuffer workspace(static_cast<std::size_t>(plan.getWorkspaceSize()));
    const v8fe::VariantPack_v8 pack = hipdnn_shim_v8_test::variantPack(
        {RELU_X_UID, RELU_Y_UID}, {x.get(), y.get()}, workspace.get());

    EXPECT_NE(cudnnBackendExecute(_handle, plan.get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_SUCCESS);
}

} // namespace

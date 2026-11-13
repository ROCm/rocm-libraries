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

#include "CudnnCompatibilityV8BuilderGraph.hpp"

namespace
{
namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace plugin_constants = hipdnn_tests::plugin_constants;
using hipdnn_compatibility_v8_test::DeviceBuffer;
using hipdnn_compatibility_v8_test::jsonEngineId;
using hipdnn_compatibility_v8_test::lastError;
using hipdnn_compatibility_v8_test::statusOf;

using hipdnn_compatibility_v8_test::floatTensorBytes;
using hipdnn_compatibility_v8_test::RELU_X_UID;
using hipdnn_compatibility_v8_test::RELU_Y_UID;
using hipdnn_compatibility_v8_test::X_DIMS;

class IntegrationCudnnCompatibilityV8BuilderEngines : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::int64_t engineCount()
    {
        const v8fe::OperationGraph_v8 graph
            = hipdnn_compatibility_v8_test::reluGraph(_handle, RELU_X_UID, RELU_Y_UID, X_DIMS);
        return graph.getEngineCount();
    }
};

TEST_F(IntegrationCudnnCompatibilityV8BuilderEngines, GoodPluginHasOneEngine)
{
    EXPECT_EQ(engineCount(), 1);
}

TEST_F(IntegrationCudnnCompatibilityV8BuilderEngines, RepeatedQueriesAgree)
{
    const v8fe::OperationGraph_v8 graph
        = hipdnn_compatibility_v8_test::reluGraph(_handle, RELU_X_UID, RELU_Y_UID, X_DIMS);
    EXPECT_EQ(graph.getEngineCount(), 1);
    EXPECT_EQ(graph.getEngineCount(), 1);
}

class IntegrationCudnnCompatibilityV8BuilderEnginesThreePlugins
    : public IntegrationCudnnCompatibilityV8BuilderEngines
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testHashedNamePluginPath()};
    }
};

TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesThreePlugins, EveryLoadedEngineCounted)
{
    EXPECT_EQ(engineCount(), 3);
}

class IntegrationCudnnCompatibilityV8BuilderEnginesNoEngines
    : public IntegrationCudnnCompatibilityV8BuilderEngines
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testNoApplicableEnginesAPluginPath()};
    }
};

TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesNoEngines, ZeroIsNotAnError)
{
    EXPECT_EQ(engineCount(), 0);
}

// Engines sorted by signed id: -34, -33, -2, then the execute-fails engine's positive id.
constexpr std::int64_t K_ENGINE_COUNT = 4;
constexpr std::int64_t K_CONTEXT_FAILS_INDEX = 0;
constexpr std::int64_t K_EXECUTE_FAILS_INDEX = 3;

class IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine
    : public hipdnn_compatibility_v8_test::ReluGraphFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testCudnnCompatibilityV8PluginPath()};
    }

    // Each config's dense index, from its plan's tag. Only the context-failing
    // engine's plan cannot be built; it reads as that engine's index.
    std::vector<std::int64_t> rankedIndices(v8fe::EngineConfigList& configs)
    {
        std::vector<std::int64_t> indices;
        indices.reserve(configs.size());
        std::size_t failures = 0;
        for(v8fe::ManagedOpaqueDescriptor& config : configs)
        {
            const cudnnStatus_t status = statusOf([&] {
                const v8fe::ExecutionPlan_v8 plan = v8fe::ExecutionPlanBuilder_v8()
                                                        .setEngineConfig(config, _graph->getTag())
                                                        .build();
                indices.push_back(hipdnn_compatibility_v8_test::tagEngineIndex(plan.getTag()));
            });
            if(status != CUDNN_STATUS_SUCCESS)
            {
                ++failures;
                indices.push_back(K_CONTEXT_FAILS_INDEX);
            }
        }
        EXPECT_EQ(failures, 1U);
        return indices;
    }
};

TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine, IndexFollowsSignedEngineId)
{
    ASSERT_EQ(_graph->getEngineCount(), K_ENGINE_COUNT);

    const std::array<std::int64_t, 3> builtIds
        = {plugin_constants::engineId<CudnnCompatibilityV8Plugin>(),
           plugin_constants::engineId<GoodPlugin>(),
           plugin_constants::engineId<ExecuteFailsPlugin>()};
    for(std::int64_t index = K_CONTEXT_FAILS_INDEX + 1; index < K_ENGINE_COUNT; ++index)
    {
        SCOPED_TRACE(index);
        const v8fe::ExecutionPlan_v8 plan = hipdnn_compatibility_v8_test::planFor(*_graph, index);
        EXPECT_EQ(hipdnn_compatibility_v8_test::tagEngineIndex(plan.getTag()), index);
        EXPECT_EQ(jsonEngineId(plan.getJsonRepresentation()),
                  std::to_string(builtIds.at(static_cast<std::size_t>(index - 1))));
    }
}

// A plan's tag names its config's dense index; the one config whose plan cannot be
// built is the context-failing engine's.
TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine, ResultsFollowHeuristicRank)
{
    v8fe::EngineConfigList configs = hipdnn_compatibility_v8_test::heuristicConfigs(*_graph);
    ASSERT_EQ(static_cast<std::int64_t>(configs.size()), K_ENGINE_COUNT);

    const std::vector<std::int64_t> ranked = rankedIndices(configs);
    std::vector<std::int64_t> indexOrder(configs.size());
    std::iota(indexOrder.begin(), indexOrder.end(), 0);
    EXPECT_TRUE(std::is_permutation(ranked.begin(), ranked.end(), indexOrder.begin()));
    EXPECT_NE(ranked, indexOrder);
}

// A, B and INSTANT each run their own policy query, which this suite's
// HIPDNN_HEUR_POLICY_ORDER overrides; every mode returns the FALLBACK engines.
TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine, EveryModeReturnsFallbackEngines)
{
    v8fe::EngineConfigList fallback = hipdnn_compatibility_v8_test::heuristicConfigs(*_graph);
    const std::vector<std::int64_t> fallbackIndices = rankedIndices(fallback);
    ASSERT_EQ(static_cast<std::int64_t>(fallbackIndices.size()), K_ENGINE_COUNT);

    for(const cudnnBackendHeurMode_t mode :
        {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_INSTANT})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        v8fe::EngineConfigList configs
            = hipdnn_compatibility_v8_test::heuristicConfigs(*_graph, mode);
        const std::vector<std::int64_t> indices = rankedIndices(configs);
        ASSERT_EQ(indices.size(), fallbackIndices.size());
        EXPECT_TRUE(std::is_permutation(indices.begin(), indices.end(), fallbackIndices.begin()));
    }
    EXPECT_EQ(_graph->getEngineCount(), K_ENGINE_COUNT);
}

TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine, ContextFailureFailsOnlyItsPlan)
{
    for(std::int64_t index = 0; index < K_ENGINE_COUNT; ++index)
    {
        SCOPED_TRACE(index);
        const cudnnStatus_t status = statusOf([&] {
            [[maybe_unused]] auto plan = hipdnn_compatibility_v8_test::planFor(*_graph, index);
        });
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

TEST_F(IntegrationCudnnCompatibilityV8BuilderEnginesMultiEngine, ExecuteFailureReported)
{
    const v8fe::ExecutionPlan_v8 plan
        = hipdnn_compatibility_v8_test::planFor(*_graph, K_EXECUTE_FAILS_INDEX);

    const DeviceBuffer x(floatTensorBytes(X_DIMS));
    const DeviceBuffer y(floatTensorBytes(X_DIMS));
    const DeviceBuffer workspace(static_cast<std::size_t>(plan.getWorkspaceSize()));
    const v8fe::VariantPack_v8 pack = hipdnn_compatibility_v8_test::variantPack(
        {RELU_X_UID, RELU_Y_UID}, {x.get(), y.get()}, workspace.get());

    EXPECT_NE(cudnnBackendExecute(_handle, plan.get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_SUCCESS);
}

} // namespace

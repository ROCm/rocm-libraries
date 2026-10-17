// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cuDNN v8 engine index, heuristic rank, and per-engine plan failures across several
// test plugins against the real hipDNN backend. The chained test heuristic policy
// reverses hipDNN's candidate order so rank and index differ. Engine queries need
// hipDNN heuristics, which query the device, so these tests skip on device-less hosts.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
using hipdnn_shim_v8_test::DeviceBuffer;
using hipdnn_shim_v8_test::GraphBuilder;
using hipdnn_shim_v8_test::jsonEngineId;
using hipdnn_shim_v8_test::lastError;
namespace plugin_constants = hipdnn_tests::plugin_constants;

constexpr std::int64_t K_X_UID = 1;
constexpr std::int64_t K_Y_UID = 2;
constexpr std::size_t K_TENSOR_BYTES = std::size_t{2} * 4 * 8 * 8 * sizeof(float);

// Engines sorted by signed id: -32, -31, -2, then the execute-fails engine's positive id.
constexpr std::int64_t K_ENGINE_COUNT = 4;
constexpr std::int64_t K_CONTEXT_FAILS_INDEX = 0;
constexpr std::int64_t K_EXECUTE_FAILS_INDEX = 3;

class IntegrationCudnnShimV8MultiEngine : public hipdnn_tests::IntegrationTestFixture
{
protected:
    static void SetUpTestSuite()
    {
        const std::array<const char*, 1> heuristicPaths
            = {plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        sPolicyOrderEnv.emplace("HIPDNN_HEUR_POLICY_ORDER",
                                plugin_constants::testGoodHeuristicPolicyName());
    }

    static void TearDownTestSuite()
    {
        sPolicyOrderEnv.reset();
        const std::array<const char*, 1> heuristicPaths
            = {plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testGoodPluginPath(),
                plugin_constants::testExecuteFailsPluginPath(),
                plugin_constants::testCudnnShimV8PluginPath()};
    }

    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        _graph = _builder->reluGraph(_handle, K_X_UID, K_Y_UID, {2, 4, 8, 8});
    }

    // Plans own plugin execution contexts, so they go before the handle that loaded the plugin.
    void TearDown() override
    {
        _builder.reset();
        IntegrationTestFixture::TearDown();
    }

    // The plan for an engine index, finalized; its finalize status is returned through status.
    cudnnBackendDescriptor_t planFor(std::int64_t engineIndex, cudnnStatus_t& status)
    {
        cudnnBackendDescriptor_t config
            = _builder->engineConfig(_builder->engine(_graph, engineIndex));
        EXPECT_EQ(cudnnBackendFinalize(config), CUDNN_STATUS_SUCCESS) << lastError();
        cudnnBackendDescriptor_t plan = _builder->plan(_handle, config);
        status = cudnnBackendFinalize(plan);
        return plan;
    }

    std::int64_t engineCount() const
    {
        return GraphBuilder::getInt64(_graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT);
    }

    static inline std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>
        sPolicyOrderEnv;

    std::unique_ptr<GraphBuilder> _builder = std::make_unique<GraphBuilder>();
    cudnnBackendDescriptor_t _graph = nullptr;
};

TEST_F(IntegrationCudnnShimV8MultiEngine, IndexFollowsSignedEngineId)
{
    ASSERT_EQ(engineCount(), K_ENGINE_COUNT);

    const std::array<std::int64_t, 3> builtIds = {plugin_constants::engineId<CudnnShimV8Plugin>(),
                                                  plugin_constants::engineId<GoodPlugin>(),
                                                  plugin_constants::engineId<ExecuteFailsPlugin>()};
    for(std::int64_t index = K_CONTEXT_FAILS_INDEX + 1; index < K_ENGINE_COUNT; ++index)
    {
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
        cudnnBackendDescriptor_t plan = planFor(index, status);
        ASSERT_EQ(status, CUDNN_STATUS_SUCCESS) << "index " << index << ": " << lastError();
        EXPECT_EQ(jsonEngineId(GraphBuilder::planJson(plan)),
                  std::to_string(builtIds.at(static_cast<std::size_t>(index - 1))))
            << "index " << index;
    }
}

TEST_F(IntegrationCudnnShimV8MultiEngine, ResultsFollowHeuristicRank)
{
    const auto configs = _builder->heuristicConfigs(_graph);
    ASSERT_EQ(static_cast<std::int64_t>(configs.size()), K_ENGINE_COUNT);

    std::vector<std::int64_t> rankedIndices;
    rankedIndices.reserve(configs.size());
    for(cudnnBackendDescriptor_t config : configs)
    {
        rankedIndices.push_back(_builder->configEngineIndex(config));
    }

    std::vector<std::int64_t> indexOrder(configs.size());
    std::iota(indexOrder.begin(), indexOrder.end(), 0);
    EXPECT_TRUE(
        std::is_permutation(rankedIndices.begin(), rankedIndices.end(), indexOrder.begin()));
    EXPECT_NE(rankedIndices, indexOrder);
}

TEST_F(IntegrationCudnnShimV8MultiEngine, ContextFailureFailsOnlyItsPlan)
{
    for(std::int64_t index = 0; index < K_ENGINE_COUNT; ++index)
    {
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
        planFor(index, status);
        if(index == K_CONTEXT_FAILS_INDEX)
        {
            EXPECT_NE(status, CUDNN_STATUS_SUCCESS);
        }
        else
        {
            EXPECT_EQ(status, CUDNN_STATUS_SUCCESS) << "index " << index << ": " << lastError();
        }
    }
}

TEST_F(IntegrationCudnnShimV8MultiEngine, ExecuteFailureReported)
{
    cudnnStatus_t status = CUDNN_STATUS_INTERNAL_ERROR;
    cudnnBackendDescriptor_t plan = planFor(K_EXECUTE_FAILS_INDEX, status);
    ASSERT_EQ(status, CUDNN_STATUS_SUCCESS) << lastError();

    const DeviceBuffer x(K_TENSOR_BYTES);
    const DeviceBuffer y(K_TENSOR_BYTES);
    const DeviceBuffer workspace(static_cast<std::size_t>(
        GraphBuilder::getInt64(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)));
    cudnnBackendDescriptor_t pack
        = _builder->variantPack({K_X_UID, K_Y_UID}, {x.get(), y.get()}, workspace.get());

    EXPECT_NE(cudnnBackendExecute(_handle, plan, pack), CUDNN_STATUS_SUCCESS);
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The raw cuDNN v8 backend flow (heuristics, engine config, plan, variant pack,
// execute) against the real hipDNN backend and the good test plugin (-2), whose
// execute is a no-op and whose plans serialize. Engine queries need hipDNN
// heuristics, which query the device, so these tests skip on device-less hosts.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
using hipdnn_shim_v8_test::DeviceBuffer;
using hipdnn_shim_v8_test::GraphBuilder;
using hipdnn_shim_v8_test::jsonEngineId;
using hipdnn_shim_v8_test::lastError;

constexpr std::int64_t K_X_UID = 1;
constexpr std::int64_t K_Y_UID = 2;
constexpr std::size_t K_TENSOR_BYTES = std::size_t{2} * 4 * 8 * 8 * sizeof(float);

// Device buffers for the graph's two tensors and a plan's workspace.
struct Buffers
{
    explicit Buffers(std::int64_t workspaceSize)
        : workspace(static_cast<std::size_t>(workspaceSize))
    {
    }

    DeviceBuffer x{K_TENSOR_BYTES};
    DeviceBuffer y{K_TENSOR_BYTES};
    DeviceBuffer workspace;
};

class IntegrationCudnnShimV8Execute : public hipdnn_tests::IntegrationTestFixture
{
protected:
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

    cudnnBackendDescriptor_t finalizedPlan(cudnnHandle_t handle)
    {
        const auto configs = _builder->heuristicConfigs(_graph);
        EXPECT_EQ(configs.size(), 1U);
        if(configs.empty())
        {
            return nullptr;
        }
        cudnnBackendDescriptor_t plan = _builder->plan(handle, configs.front());
        EXPECT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_SUCCESS) << lastError();
        return plan;
    }

    cudnnBackendDescriptor_t variantPack(const Buffers& buffers)
    {
        return _builder->variantPack(
            {K_X_UID, K_Y_UID}, {buffers.x.get(), buffers.y.get()}, buffers.workspace.get());
    }

    static std::int64_t workspaceSize(cudnnBackendDescriptor_t plan)
    {
        return GraphBuilder::getInt64(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
    }

    std::unique_ptr<GraphBuilder> _builder = std::make_unique<GraphBuilder>();
    cudnnBackendDescriptor_t _graph = nullptr;
};

TEST_F(IntegrationCudnnShimV8Execute, HeuristicPlanExecutes)
{
    cudnnBackendDescriptor_t plan = finalizedPlan(_handle);
    ASSERT_NE(plan, nullptr);
    const std::int64_t workspace = workspaceSize(plan);
    ASSERT_GE(workspace, 0);

    const Buffers buffers(workspace);
    EXPECT_EQ(cudnnBackendExecute(_handle, plan, variantPack(buffers)), CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8Execute, GlobalIndexPlanExecutes)
{
    cudnnBackendDescriptor_t config = _builder->engineConfig(_builder->engine(_graph, 0));
    ASSERT_EQ(cudnnBackendFinalize(config), CUDNN_STATUS_SUCCESS) << lastError();
    cudnnBackendDescriptor_t plan = _builder->plan(_handle, config);
    ASSERT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_SUCCESS) << lastError();

    const Buffers buffers(workspaceSize(plan));
    EXPECT_EQ(cudnnBackendExecute(_handle, plan, variantPack(buffers)), CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8Execute, JsonCarriesEngineAndPayload)
{
    cudnnBackendDescriptor_t plan = finalizedPlan(_handle);
    ASSERT_NE(plan, nullptr);

    const std::string json = GraphBuilder::planJson(plan);
    EXPECT_NE(json.find("\"engine\""), std::string::npos) << json;
    EXPECT_EQ(jsonEngineId(json),
              std::to_string(hipdnn_tests::plugin_constants::engineId<GoodPlugin>()))
        << json;
    EXPECT_NE(json.find("\"payload\""), std::string::npos) << json;
}

TEST_F(IntegrationCudnnShimV8Execute, LoadedJsonPlanExecutes)
{
    cudnnBackendDescriptor_t built = finalizedPlan(_handle);
    ASSERT_NE(built, nullptr);
    const std::string json = GraphBuilder::planJson(built);

    cudnnBackendDescriptor_t loaded = _builder->loadedPlan(_handle, json);
    ASSERT_EQ(cudnnBackendFinalize(loaded), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(workspaceSize(loaded), workspaceSize(built));
    EXPECT_EQ(jsonEngineId(GraphBuilder::planJson(loaded)), jsonEngineId(json));

    const Buffers buffers(workspaceSize(loaded));
    EXPECT_EQ(cudnnBackendExecute(_handle, loaded, variantPack(buffers)), CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

class IntegrationCudnnShimV8ExecuteForeignHandle : public IntegrationCudnnShimV8Execute
{
protected:
    void SetUp() override
    {
        IntegrationCudnnShimV8Execute::SetUp();
        if(IsSkipped())
        {
            return;
        }
        ASSERT_EQ(cudnnCreate(&_otherHandle), CUDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        _builder.reset();
        if(_otherHandle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_otherHandle), CUDNN_STATUS_SUCCESS);
        }
        IntegrationCudnnShimV8Execute::TearDown();
    }

    cudnnHandle_t _otherHandle = nullptr;
};

TEST_F(IntegrationCudnnShimV8ExecuteForeignHandle, ExecuteRejected)
{
    cudnnBackendDescriptor_t plan = finalizedPlan(_handle);
    ASSERT_NE(plan, nullptr);
    const Buffers buffers(workspaceSize(plan));
    cudnnBackendDescriptor_t pack = variantPack(buffers);

    EXPECT_EQ(cudnnBackendExecute(_otherHandle, plan, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, plan, pack), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8ExecuteForeignHandle, PlanFinalizeRejected)
{
    const auto configs = _builder->heuristicConfigs(_graph);
    ASSERT_EQ(configs.size(), 1U);
    cudnnBackendDescriptor_t plan = _builder->plan(_otherHandle, configs.front());
    EXPECT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8ExecuteForeignHandle, LoadedPlanBindsItsOwnHandle)
{
    cudnnBackendDescriptor_t built = finalizedPlan(_handle);
    ASSERT_NE(built, nullptr);

    // A JSON-loaded plan has no graph; it executes on the handle it was loaded with.
    cudnnBackendDescriptor_t loaded
        = _builder->loadedPlan(_otherHandle, GraphBuilder::planJson(built));
    ASSERT_EQ(cudnnBackendFinalize(loaded), CUDNN_STATUS_SUCCESS) << lastError();

    const Buffers buffers(workspaceSize(loaded));
    cudnnBackendDescriptor_t pack = variantPack(buffers);
    EXPECT_EQ(cudnnBackendExecute(_handle, loaded, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_otherHandle, loaded, pack), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

} // namespace

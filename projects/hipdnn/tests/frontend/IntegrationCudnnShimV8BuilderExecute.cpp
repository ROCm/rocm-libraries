// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v0.x builder flow (heuristics, engine config, plan, variant pack) and
// cudnnBackendExecute against the real hipDNN backend and the good test plugin (-2),
// whose execute is a no-op and whose plans serialize. Engine queries need hipDNN
// heuristics, which query the device, so these tests skip on device-less hosts.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8BuilderGraph.hpp"

namespace
{
namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;
using hipdnn_shim_v8_test::DeviceBuffer;
using hipdnn_shim_v8_test::jsonEngineId;
using hipdnn_shim_v8_test::lastError;

constexpr std::int64_t K_X_UID = 1;
constexpr std::int64_t K_Y_UID = 2;
const std::vector<std::int64_t> K_DIMS = {2, 4, 8, 8};
constexpr std::size_t K_TENSOR_BYTES = std::size_t{2} * 4 * 8 * 8 * sizeof(float);

// Device buffers for the graph's two tensors and a plan's workspace.
struct Buffers
{
    explicit Buffers(std::int64_t workspaceSize)
        : workspace(static_cast<std::size_t>(workspaceSize))
    {
    }

    v8fe::VariantPack_v8 pack() const
    {
        return hipdnn_shim_v8_test::variantPack(
            {K_X_UID, K_Y_UID}, {x.get(), y.get()}, workspace.get());
    }

    DeviceBuffer x{K_TENSOR_BYTES};
    DeviceBuffer y{K_TENSOR_BYTES};
    DeviceBuffer workspace;
};

class IntegrationCudnnShimV8BuilderExecute : public hipdnn_tests::IntegrationTestFixture
{
protected:
    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        _graph.emplace(hipdnn_shim_v8_test::reluGraph(_handle, K_X_UID, K_Y_UID, K_DIMS));
    }

    // The graph owns plugin state, so it goes before the handle that loaded the plugin.
    void TearDown() override
    {
        _graph.reset();
        IntegrationTestFixture::TearDown();
    }

    // The plan of the one engine the heuristics return.
    v8fe::ExecutionPlan_v8 heuristicPlan()
    {
        v8fe::EngineConfigList configs = hipdnn_shim_v8_test::heuristicConfigs(*_graph);
        EXPECT_EQ(configs.size(), 1U);
        return v8fe::ExecutionPlanBuilder_v8()
            .setEngineConfig(configs.front(), _graph->getTag())
            .build();
    }

    static cudnnStatus_t execute(cudnnHandle_t handle,
                                 const v8fe::ExecutionPlan_v8& plan,
                                 const v8fe::VariantPack_v8& pack)
    {
        return cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc());
    }

    std::optional<v8fe::OperationGraph_v8> _graph;
};

TEST_F(IntegrationCudnnShimV8BuilderExecute, HeuristicPlanExecutes)
{
    const v8fe::ExecutionPlan_v8 plan = heuristicPlan();
    ASSERT_GE(plan.getWorkspaceSize(), 0);

    const Buffers buffers(plan.getWorkspaceSize());
    EXPECT_EQ(execute(_handle, plan, buffers.pack()), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8BuilderExecute, GlobalIndexPlanExecutes)
{
    const v8fe::ExecutionPlan_v8 plan = hipdnn_shim_v8_test::planFor(*_graph, 0);

    const Buffers buffers(plan.getWorkspaceSize());
    EXPECT_EQ(execute(_handle, plan, buffers.pack()), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8BuilderExecute, JsonCarriesEngineAndPayload)
{
    const std::string json = heuristicPlan().getJsonRepresentation();

    EXPECT_NE(json.find("\"engine\""), std::string::npos) << json;
    EXPECT_EQ(jsonEngineId(json),
              std::to_string(hipdnn_tests::plugin_constants::engineId<GoodPlugin>()))
        << json;
    EXPECT_NE(json.find("\"payload\""), std::string::npos) << json;
}

TEST_F(IntegrationCudnnShimV8BuilderExecute, LoadedJsonPlanExecutes)
{
    const v8fe::ExecutionPlan_v8 built = heuristicPlan();
    const std::string json = built.getJsonRepresentation();

    const v8fe::ExecutionPlan_v8 loaded
        = v8fe::ExecutionPlanBuilder_v8().setHandle(_handle).loadFromJson(json);
    EXPECT_EQ(loaded.getWorkspaceSize(), built.getWorkspaceSize());
    EXPECT_EQ(jsonEngineId(loaded.getJsonRepresentation()), jsonEngineId(json));

    const Buffers buffers(loaded.getWorkspaceSize());
    EXPECT_EQ(execute(_handle, loaded, buffers.pack()), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8BuilderExecute, NullHandleOrDescriptorsRejected)
{
    const v8fe::ExecutionPlan_v8 plan = heuristicPlan();
    const Buffers buffers(plan.getWorkspaceSize());
    const v8fe::VariantPack_v8 pack = buffers.pack();

    EXPECT_EQ(execute(nullptr, plan, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, nullptr, nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, plan.get_raw_desc(), nullptr), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8BuilderExecute, SwappedDescriptorsRejected)
{
    const v8fe::ExecutionPlan_v8 plan = heuristicPlan();
    const Buffers buffers(plan.getWorkspaceSize());
    const v8fe::VariantPack_v8 pack = buffers.pack();

    EXPECT_EQ(cudnnBackendExecute(_handle, pack.get_raw_desc(), plan.get_raw_desc()),
              CUDNN_STATUS_BAD_PARAM);
}

// Executing a plan on a handle other than the one it was built or loaded on.
class IntegrationCudnnShimV8BuilderExecuteForeignHandle
    : public IntegrationCudnnShimV8BuilderExecute
{
protected:
    // cudnnBackendExecute passes straight through, so hipDNN runs the plan on the given handle.
    static constexpr cudnnStatus_t K_FOREIGN_HANDLE_STATUS = CUDNN_STATUS_SUCCESS;

    void SetUp() override
    {
        IntegrationCudnnShimV8BuilderExecute::SetUp();
        if(IsSkipped())
        {
            return;
        }
        ASSERT_EQ(cudnnCreate(&_otherHandle), CUDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_otherHandle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_otherHandle), CUDNN_STATUS_SUCCESS);
        }
        IntegrationCudnnShimV8BuilderExecute::TearDown();
    }

    // The good test plugin's execute is a no-op, so its correct output is the y buffer as the
    // caller left it: a foreign-handle run must neither fail nor touch the tensors.
    static void fill(const DeviceBuffer& buffer, int byte)
    {
        ASSERT_EQ(hipMemset(buffer.get(), byte, K_TENSOR_BYTES), hipSuccess);
    }

    static bool holds(const DeviceBuffer& buffer, unsigned char byte)
    {
        std::vector<unsigned char> host(K_TENSOR_BYTES);
        EXPECT_EQ(hipMemcpy(host.data(), buffer.get(), host.size(), hipMemcpyDeviceToHost),
                  hipSuccess);
        return std::all_of(
            host.begin(), host.end(), [byte](unsigned char value) { return value == byte; });
    }

    cudnnHandle_t _otherHandle = nullptr;
};

TEST_F(IntegrationCudnnShimV8BuilderExecuteForeignHandle, BuiltPlan)
{
    const v8fe::ExecutionPlan_v8 plan = heuristicPlan();
    const Buffers buffers(plan.getWorkspaceSize());
    const v8fe::VariantPack_v8 pack = buffers.pack();
    fill(buffers.x, 0x11);
    fill(buffers.y, 0x5A);

    EXPECT_EQ(execute(_otherHandle, plan, pack), K_FOREIGN_HANDLE_STATUS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_TRUE(holds(buffers.x, 0x11));
    EXPECT_TRUE(holds(buffers.y, 0x5A));
    EXPECT_EQ(execute(_handle, plan, pack), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(IntegrationCudnnShimV8BuilderExecuteForeignHandle, LoadedPlan)
{
    const std::string json = heuristicPlan().getJsonRepresentation();
    const v8fe::ExecutionPlan_v8 loaded
        = v8fe::ExecutionPlanBuilder_v8().setHandle(_otherHandle).loadFromJson(json);
    const Buffers buffers(loaded.getWorkspaceSize());
    const v8fe::VariantPack_v8 pack = buffers.pack();
    fill(buffers.x, 0x11);
    fill(buffers.y, 0x5A);

    EXPECT_EQ(execute(_handle, loaded, pack), K_FOREIGN_HANDLE_STATUS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_TRUE(holds(buffers.x, 0x11));
    EXPECT_TRUE(holds(buffers.y, 0x5A));
    EXPECT_EQ(execute(_otherHandle, loaded, pack), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

} // namespace

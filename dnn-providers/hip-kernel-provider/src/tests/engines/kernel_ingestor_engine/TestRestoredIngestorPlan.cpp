// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestRestoredIngestorPlan.cpp
 * @brief RestoredIngestorPlan: the workspace it reports, the workspace it requires, and the
 *        launch it hands to its dispatch handler. No device is needed.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;

// Counts its launches and records the last prepared dispatch and workspace it received.
class CountingHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const Handle& /*handle*/,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* workspace) const override
    {
        ++_launches;
        _lastPrepared = &prepared;
        _lastWorkspace = workspace;
    }

    int launches() const
    {
        return _launches.load();
    }

    const PreparedDispatch* lastPrepared() const
    {
        return _lastPrepared.load();
    }

    void* lastWorkspace() const
    {
        return _lastWorkspace.load();
    }

private:
    mutable std::atomic<int> _launches{0};
    mutable std::atomic<const PreparedDispatch*> _lastPrepared{nullptr};
    mutable std::atomic<void*> _lastWorkspace{nullptr};
};

TEST(TestRestoredIngestorPlan, ReportsThePayloadWorkspaceAndLaunchesThroughItsHandler)
{
    constexpr size_t WORKSPACE_BYTES = 1024;
    const CountingHandler handler;
    auto prepared = std::make_unique<PreparedDispatch>();
    const PreparedDispatch* const preparedAddress = prepared.get();
    const RestoredIngestorPlan<Handle> plan(
        handler, std::move(prepared), WORKSPACE_BYTES, "test plan");
    const hipdnn_plugin_sdk::IPlan<Handle>& asPlan = plan;
    const Handle handle;

    EXPECT_EQ(asPlan.getWorkspaceSize(handle), WORKSPACE_BYTES);
    EXPECT_EQ(asPlan.saveablePlan(), nullptr);

    serialization::expectPluginException([&]() { asPlan.execute(handle, nullptr, 0, nullptr); },
                                         HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                         "",
                                         "test plan requires 1024 workspace bytes");
    EXPECT_EQ(handler.launches(), 0);

    // The handler only records the pointer, so any address stands in for a workspace.
    int workspace = 0;
    asPlan.execute(handle, nullptr, 0, &workspace);
    EXPECT_EQ(handler.launches(), 1);
    EXPECT_EQ(handler.lastPrepared(), preparedAddress);
    EXPECT_EQ(handler.lastWorkspace(), &workspace);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

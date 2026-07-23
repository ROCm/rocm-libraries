// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "RockeClientContext.hpp"
#include "RockeClientHandle.hpp"
#include "RockeClientSettings.hpp"
#include "dispatcher/AotCatalog.hpp"
#include "dispatcher/RockeClientDispatcher.hpp"

namespace rocke_client
{

class RockeClientEngine
    : public hipdnn_plugin_sdk::IEngine<RockeClientHandle, RockeClientSettings, RockeClientContext>
{
public:
    // Production engine: selects over the installed AOT catalog (empty until the
    // kpack producer lands, so today it declines every graph).
    RockeClientEngine();

    // Test/advanced construction: inject a catalog directly.
    explicit RockeClientEngine(dispatcher::AotCatalog catalog);

    int64_t id() const override;

    bool isApplicable(
        RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(RockeClientHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const RockeClientHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        RockeClientContext& executionContext) const override;

private:
    dispatcher::RockeClientDispatcher _dispatcher;
};

} // namespace rocke_client

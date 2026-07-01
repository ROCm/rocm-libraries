// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "RockeClientContext.hpp"
#include "RockeClientHandle.hpp"
#include "RockeClientSettings.hpp"

namespace rocke_client
{

class RockeClientEngine
    : public hipdnn_plugin_sdk::IEngine<RockeClientHandle, RockeClientSettings, RockeClientContext>
{
public:
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
};

} // namespace rocke_client

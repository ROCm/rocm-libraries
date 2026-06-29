// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "RocKEContext.hpp"
#include "RocKEHandle.hpp"
#include "RocKESettings.hpp"

namespace rocke_client
{

class RocKEEngine : public hipdnn_plugin_sdk::IEngine<RocKEHandle, RocKESettings, RocKEContext>
{
public:
    int64_t id() const override;

    bool isApplicable(
        RocKEHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(RocKEHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const RocKEHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const RocKEHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        RocKEContext& executionContext) const override;
};

} // namespace rocke_client

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "engines/RockeClientEngine.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>

namespace rocke_client
{

int64_t RockeClientEngine::id() const
{
    return hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID;
}

bool RockeClientEngine::isApplicable(
    RockeClientHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return false;
}

void RockeClientEngine::getDetails(
    RockeClientHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(builder, id(), nullptr);
    builder.Finish(engineDetails);

    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t RockeClientEngine::getMaxWorkspaceSize(
    const RockeClientHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "rocke-client skeleton does not provide applicable execution plans");
}

void RockeClientEngine::initializeExecutionContext(
    const RockeClientHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    RockeClientContext& /*executionContext*/) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "rocke-client skeleton does not provide applicable execution plans");
}

} // namespace rocke_client

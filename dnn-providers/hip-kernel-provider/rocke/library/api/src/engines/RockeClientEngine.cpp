// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "engines/RockeClientEngine.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>
#include <utility>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "dispatcher/AotCatalog.hpp"
#include "plans/RockeClientPlan.hpp"

namespace rocke_client
{

RockeClientEngine::RockeClientEngine() = default;

RockeClientEngine::RockeClientEngine(dispatcher::AotCatalog catalog)
    : _dispatcher(std::move(catalog))
{
}

int64_t RockeClientEngine::id() const
{
    return hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID;
}

bool RockeClientEngine::isApplicable(
    RockeClientHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Graph accept/reject: true only if the installed kpack AOT catalog holds an
    // instance that can serve this graph on the handle's stream device.
    return _dispatcher.isApplicable(handle, opGraph);
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
    const RockeClientHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    const auto instance = _dispatcher.selectInstance(handle, opGraph);
    if(!instance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            "rocke-client: no AOT instance matches this graph");
    }

    return 0;
}

void RockeClientEngine::initializeExecutionContext(
    const RockeClientHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    RockeClientContext& executionContext) const
{
    const auto instance = _dispatcher.selectInstance(handle, opGraph);
    if(!instance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            "rocke-client: no AOT instance matches this graph");
    }

    executionContext.setExecutionSettings(RockeClientSettings{});
    executionContext.setPlan(std::make_unique<RockeClientPlan>(*instance, opGraph, handle));
}

} // namespace rocke_client

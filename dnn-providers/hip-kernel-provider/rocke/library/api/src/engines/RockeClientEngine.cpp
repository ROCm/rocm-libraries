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

namespace rocke_client
{

RockeClientEngine::RockeClientEngine()
    : _dispatcher(dispatcher::AotCatalog::loadDefault())
{
}

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
    // Graph accept/reject: true only if the AOT catalog holds an instance that
    // can serve this graph. Phase 1: the catalog is empty, so this is false.
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

    // TODO(kpack): return the workspace size from the winning instance's sidecar
    // launch metadata once PR #8866's kpack packaging lands. No AOT plan can be
    // built yet, so decline rather than report a bogus size.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "rocke-client: AOT plan construction (kpack) not yet implemented");
}

void RockeClientEngine::initializeExecutionContext(
    const RockeClientHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    RockeClientContext& /*executionContext*/) const
{
    const auto instance = _dispatcher.selectInstance(handle, opGraph);
    if(!instance.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            "rocke-client: no AOT instance matches this graph");
    }

    // A winning instance was selected -- this is the plan-construction seam.
    // TODO(kpack): once PR #8866's kpack packaging lands, replace this decline with:
    //   1. resolve kernel_id (cache_key) + launch metadata from the instance's sidecar;
    //   2. load the pre-built HSACO from the kpack (hipModuleLoad/hipModuleGetFunction);
    //   3. evaluate the symbolic grid_formula -> concrete grid[3];
    //   4. executionContext.setExecutionSettings(RockeClientSettings{});
    //   5. executionContext.setPlan(std::make_unique<RockeClientPlan>(module, params, launchMeta));
    HIPDNN_PLUGIN_LOG_WARN("rocke-client selected AOT instance '"
                           << instance->name
                           << "' but plan construction (kpack) is not implemented yet");
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "rocke-client: AOT plan construction (kpack) not yet implemented");
}

} // namespace rocke_client

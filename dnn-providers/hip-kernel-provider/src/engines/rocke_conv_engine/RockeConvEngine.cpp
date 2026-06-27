// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RockeConvEngine.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace rocke_conv_engine
{

RockeConvEngine::RockeConvEngine() = default;

void RockeConvEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder)
{
    _planBuilders.emplace_back(std::move(planBuilder));
}

int64_t RockeConvEngine::id() const
{
    return staticId();
}

int64_t RockeConvEngine::staticId()
{
    return hipdnn_data_sdk::utilities::ROCKE_CONV_ENGINE_ID;
}

bool RockeConvEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void RockeConvEngine::getDetails(
    Handle& handle,
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
    auto* dataPtr = detachedBuffer->data();
    handle.storeEngineDetailsDetachedBuffer(dataPtr, std::move(detachedBuffer));
}

size_t RockeConvEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return pb->getMaxWorkspaceSize(handle, opGraph, Settings{});
        }
    }
    HIPDNN_PLUGIN_LOG_ERROR("RockeConvEngine::getMaxWorkspaceSize: no applicable plan builder");
    return 0;
}

void RockeConvEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    executionContext.setExecutionSettings(Settings{});

    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            pb->buildPlan(handle, opGraph, engineConfig, executionContext);
            return;
        }
    }
    HIPDNN_PLUGIN_LOG_ERROR(
        "RockeConvEngine::initializeExecutionContext: no applicable plan builder");
}

} // namespace rocke_conv_engine

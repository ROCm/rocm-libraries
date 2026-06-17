// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipFlash2Engine.hpp"

// Full IPlanBuilder implementation (supersedes stub HipFlash2FwdPlanBuilder.hpp)
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipFlash2FwdPlanBuilder_v2.hpp"

namespace hip_flash2_engine
{

HipFlash2Engine::HipFlash2Engine(int64_t engineId)
    : _id(engineId)
{
}

void HipFlash2Engine::addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder)
{
    _planBuilders.emplace_back(std::move(planBuilder));
}

int64_t HipFlash2Engine::id() const
{
    return _id;
}

bool HipFlash2Engine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
            return true;
    }
    return false;
}

void HipFlash2Engine::getDetails(
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

size_t HipFlash2Engine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    // Flash-Attention 2 V7 uses only registers and LDS — no external workspace.
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
            return pb->getMaxWorkspaceSize(handle, opGraph, Settings{});
    }
    HIPDNN_PLUGIN_LOG_ERROR("HipFlash2Engine::getMaxWorkspaceSize: no applicable plan builder");
    return 0;
}

void HipFlash2Engine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    // Initialize execution settings first (required by the context before buildPlan)
    executionContext.setExecutionSettings(Settings{});

    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            // buildPlan loads the .co, extracts params, and stores a HipFlash2FwdPlan
            // in executionContext via executionContext.setPlan(...)
            pb->buildPlan(handle, opGraph, engineConfig, executionContext);
            return;
        }
    }

    HIPDNN_PLUGIN_LOG_ERROR(
        "HipFlash2Engine::initializeExecutionContext: no applicable plan builder found");
}

} // namespace hip_flash2_engine

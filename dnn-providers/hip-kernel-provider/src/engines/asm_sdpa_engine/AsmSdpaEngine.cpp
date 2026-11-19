// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace asm_sdpa_engine
{

AsmSdpaEngine::AsmSdpaEngine() = default;

void AsmSdpaEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder)
{
    _planBuilders.emplace_back(std::move(planBuilder));
}

int64_t AsmSdpaEngine::id() const
{
    return staticId();
}

int64_t AsmSdpaEngine::staticId()
{
    return hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_ID;
}

bool AsmSdpaEngine::isApplicable(
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

void AsmSdpaEngine::getDetails(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int32_t> behaviorNotes{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(
        builder, id(), nullptr, &behaviorNotes);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    auto* dataPtr = detachedBuffer->data();
    handle.storeEngineDetailsDetachedBuffer(dataPtr, std::move(detachedBuffer));
}

size_t AsmSdpaEngine::getMaxWorkspaceSize(
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

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "AsmSdpaEngine::getMaxWorkspaceSize: no plan builder applies to the graph");
}

void AsmSdpaEngine::initializeExecutionContext(
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

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "AsmSdpaEngine::initializeExecutionContext: no plan builder applies to the graph");
}

} // namespace asm_sdpa_engine

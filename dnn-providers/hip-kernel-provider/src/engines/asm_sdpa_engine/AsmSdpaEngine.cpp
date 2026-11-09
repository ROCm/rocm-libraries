// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

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

const IPlanBuilder* AsmSdpaEngine::findApplicablePlanBuilder(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return pb.get();
        }
    }
    return nullptr;
}

bool AsmSdpaEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return findApplicablePlanBuilder(handle, opGraph) != nullptr;
}

void AsmSdpaEngine::getDetails(Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    // Knobs come from the builder that serves the graph, so a forward graph never
    // advertises the backward builder's knobs.
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;
    if(const auto* pb = findApplicablePlanBuilder(handle, opGraph))
    {
        for(const auto& knobT : pb->getCustomKnobs(handle, opGraph))
        {
            knobsVector.push_back(
                hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobT));
        }
    }
    auto knobs = builder.CreateVector(knobsVector);

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, id(), knobs);
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
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    const auto* pb = findApplicablePlanBuilder(handle, opGraph);
    if(pb == nullptr)
    {
        HIPDNN_PLUGIN_LOG_ERROR("AsmSdpaEngine::getMaxWorkspaceSize: no supporting engine found");
        return 0;
    }

    // Size with the same knob settings initializeExecutionContext builds the plan
    // with, so the workspace matches the plan the engine config selects.
    Settings executionSettings;
    pb->initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
    return pb->getMaxWorkspaceSize(handle, opGraph, executionSettings);
}

void AsmSdpaEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    const auto* pb = findApplicablePlanBuilder(handle, opGraph);
    if(pb == nullptr)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "AsmSdpaEngine::initializeExecutionContext: no supporting engine found");
        return;
    }

    // buildPlan reads the knob settings from the context, so they are stored first.
    Settings executionSettings;
    pb->initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
    executionContext.setExecutionSettings(executionSettings);
    pb->buildPlan(handle, opGraph, engineConfig, executionContext);
}

} // namespace asm_sdpa_engine

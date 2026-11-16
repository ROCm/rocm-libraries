// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaEngine.hpp"
#include "core/Utils.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <vector>

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
    // Knobs come from the builder that serves the graph, so a forward graph never
    // advertises the backward builder's knobs.
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;
    if(const auto* pb = findApplicablePlanBuilder(handle, opGraph))
    {
        knobs = pb->getCustomKnobs(handle, opGraph);
    }
    const std::vector<int32_t> behaviorNotes{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    hip_kernel_provider::core::utils::storeEngineDetails(
        handle, id(), knobs, behaviorNotes, detailsOut);
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

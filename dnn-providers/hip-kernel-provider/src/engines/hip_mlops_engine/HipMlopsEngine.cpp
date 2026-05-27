// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipMlopsEngine.hpp"
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

namespace hip_kernel_provider
{

HipMlopsEngine::HipMlopsEngine(int64_t id)
    : _id(id)
{
}

int64_t HipMlopsEngine::id() const
{
    return _id;
}

static void initializeHipKernelSettings(
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings)
{
}

bool HipMlopsEngine::isApplicable(
    HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void HipMlopsEngine::getDetails(HipKernelHandle& handle,
                                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                                hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;

    // Collect custom knobs from plan builders
    for(const auto& planBuilder : _planBuilders)
    {
        auto customKnobs = planBuilder->getCustomKnobs(handle, opGraph);

        if(customKnobs.empty())
        {
            continue;
        }

        for(const auto& knobT : customKnobs)
        {
            auto knobOffset = hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobT);
            knobsVector.push_back(knobOffset);
        }

        // Only one plan builder should be applicable for a given graph and return custom knobs.
        // Stop after finding the first one to avoid duplicates.
        break;
    }

    auto knobs = builder.CreateVector(knobsVector);

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t HipMlopsEngine::getMaxWorkspaceSize(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    HipKernelSettings baseExecutionSettings;
    initializeHipKernelSettings(engineConfig, baseExecutionSettings);
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            HipKernelSettings executionSettings = baseExecutionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }
    return workspaceSize;
}

void HipMlopsEngine::initializeExecutionContext(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipKernelContext& executionContext) const
{
    HipKernelSettings executionSettings;
    initializeHipKernelSettings(engineConfig, executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            break;
        }
    }

    executionContext.setExecutionSettings(executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
            break;
        }
    }
}

void HipMlopsEngine::addPlanBuilder(
    std::unique_ptr<
        hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>>
        planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}

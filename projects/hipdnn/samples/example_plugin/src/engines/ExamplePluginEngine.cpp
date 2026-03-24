// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExamplePluginEngine.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace example_plugin
{

ExamplePluginEngine::ExamplePluginEngine(int64_t id)
    : _id(id)
{
}

int64_t ExamplePluginEngine::id() const
{
    return _id;
}

bool ExamplePluginEngine::isApplicable(
    ExamplePluginHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void ExamplePluginEngine::getDetails(ExamplePluginHandle& handle,
                                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                                     hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    // Collect custom knobs from applicable plan builders
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>> knobsVector;

    for(const auto& planBuilder : _planBuilders)
    {
        auto customKnobs = planBuilder->getCustomKnobs(handle, opGraph);
        if(customKnobs.empty())
        {
            continue;
        }

        for(const auto& knobT : customKnobs)
        {
            auto knobOffset = hipdnn_data_sdk::data_objects::Knob::Pack(builder, &knobT);
            knobsVector.push_back(knobOffset);
        }

        // Only one plan builder should be applicable -- stop after finding the first
        break;
    }

    auto knobs = builder.CreateVector(knobsVector);
    auto engineDetails = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);

    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t ExamplePluginEngine::getMaxWorkspaceSize(
    const ExamplePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    ExamplePluginSettings executionSettings;
    size_t workspaceSize = 0;

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }

    return workspaceSize;
}

void ExamplePluginEngine::initializeExecutionContext(
    const ExamplePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    ExamplePluginContext& executionContext) const
{
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            ExamplePluginSettings executionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            executionContext.setExecutionSettings(executionSettings);
            planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
            return;
        }
    }
}

void ExamplePluginEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                    ExamplePluginSettings,
                                                    ExamplePluginContext>> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

} // namespace example_plugin

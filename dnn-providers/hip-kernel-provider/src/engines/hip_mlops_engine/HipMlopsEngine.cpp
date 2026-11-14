// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipMlopsEngine.hpp"
#include "core/Utils.hpp"
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

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
    [[maybe_unused]] Settings& executionSettings)
{
}

bool HipMlopsEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[HipMlopsEngine::isApplicable] ";

    const auto& tensorMap = opGraph.getTensorMap();
    HIP_KERNEL_RETURN_FALSE_IF(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap),
                               "Ragged tensors are not supported");
    HIP_KERNEL_RETURN_FALSE_IF(
        !hip_kernel_provider::core::utils::hasOnly16ByteAlignedTensors(tensorMap),
        "All tensors must be aligned to a multiple of 16 bytes");

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

void HipMlopsEngine::getDetails(Handle& handle,
                                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                                hipdnnPluginConstData_t& detailsOut) const
{
    // Only one plan builder should be applicable for a given graph and return custom knobs.
    // Take the first one's to avoid duplicates.
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;
    for(const auto& planBuilder : _planBuilders)
    {
        knobs = planBuilder->getCustomKnobs(handle, opGraph);
        if(!knobs.empty())
        {
            break;
        }
    }

    const std::vector<int32_t> behaviorNotes{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    core::utils::storeEngineDetails(handle, _id, knobs, behaviorNotes, detailsOut);
}

size_t HipMlopsEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    Settings baseExecutionSettings;
    initializeHipKernelSettings(engineConfig, baseExecutionSettings);
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            Settings executionSettings = baseExecutionSettings;
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
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    Settings executionSettings;
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
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipFlash2Engine.hpp"

#include "HipFlash2FwdPlanBuilder.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

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
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
            return true;
    }
    return false;
}

void HipFlash2Engine::getDetails(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    hipdnnPluginConstData_t& detailsOut) const
{
    // Return engine name and ID as detail
    static const auto idVal = staticId();
    detailsOut.data = &idVal;
    detailsOut.sizeInBytes = sizeof(idVal);
}

size_t HipFlash2Engine::getMaxWorkspaceSize(
    const Handle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    // Flash-Attention 2 uses only registers and LDS — no global workspace needed
    return 0;
}

void HipFlash2Engine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionContext(
                handle, opGraph, engineConfig, executionContext);
            return;
        }
    }
}

} // namespace hip_flash2_engine

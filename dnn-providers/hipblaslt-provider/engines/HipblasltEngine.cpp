// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltEngine.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

namespace hipblaslt_plugin
{

HipblasltEngine::HipblasltEngine(int64_t id)
    : _id(id)
{
}

int64_t HipblasltEngine::id() const
{
    return _id;
}

bool HipblasltEngine::isApplicable(
    HipdnnEnginePluginHandle& handle,
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

void HipblasltEngine::getDetails(HipdnnEnginePluginHandle& handle,
                                 hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t HipblasltEngine::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            workspaceSize = std::max(workspaceSize, planBuilder->getWorkspaceSize(handle, opGraph));
        }
    }
    return workspaceSize;
}

void HipblasltEngine::initializeExecutionContext(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, executionContext);
            break;
        }
    }
}

void HipblasltEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

} // namespace hipblaslt_plugin

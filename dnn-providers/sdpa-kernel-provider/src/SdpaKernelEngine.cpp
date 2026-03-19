// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelEngine.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <ranges>

namespace sdpa_kernel_provider
{

void SdpaKernelEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder)
{
    _planBuilders.emplace_back(std::move(planBuilder));
}

int64_t SdpaKernelEngine::id() const
{
    return staticId();
}

int64_t SdpaKernelEngine::staticId()
{
    static int64_t s_cachedId = hipdnn_data_sdk::utilities::engineNameToId(engineName());
    return s_cachedId;
}

bool SdpaKernelEngine::isApplicable(
    SdpaKernelHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return std::ranges::any_of(_planBuilders,
                               [&](const auto& pb) { return pb->isApplicable(handle, opGraph); });
}

void SdpaKernelEngine::getDetails(SdpaKernelHandle& handle,
                                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                                  hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto engineDetails
        = hipdnn_data_sdk::data_objects::CreateEngineDetailsDirect(builder, id(), nullptr);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detachedBuffer->data(), std::move(detachedBuffer));
}

size_t SdpaKernelEngine::getMaxWorkspaceSize(
    const SdpaKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            return pb->getMaxWorkspaceSize(handle, opGraph, SdpaKernelSettings(engineConfig));
        }
    }

    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelEngine::getMaxWorkspaceSize: no supporting engine found");
    return 0;
}

void SdpaKernelEngine::initializeExecutionContext(
    const SdpaKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    SdpaKernelContext& executionContext) const
{
    for(const auto& pb : _planBuilders)
    {
        if(pb->isApplicable(handle, opGraph))
        {
            pb->buildPlan(handle, opGraph, engineConfig, executionContext);
            return;
        }
    }

    HIPDNN_PLUGIN_LOG_ERROR(
        "SdpaKernelEngine::initializeExecutionContext: no supporting engine found");
}

}

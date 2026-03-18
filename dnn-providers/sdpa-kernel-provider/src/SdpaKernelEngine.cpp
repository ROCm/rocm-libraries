// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelEngine.hpp"
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace sdpa_kernel_provider
{

int64_t SdpaKernelEngine::id() const
{
    return hipdnn_data_sdk::utilities::engineNameToId(engineName());
}

bool SdpaKernelEngine::isApplicable(
    SdpaKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto& nodeWrappers = opGraph.nodeWrappers();

    if(nodeWrappers.size() != 1
       || nodeWrappers.front()->attributesType()
              != hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes)
    {
        return false;
    }

    // TODO: Add more expansive checks
    HIPDNN_PLUGIN_LOG_WARN("SdpaKernelEngine::isApplicable not fully implemented");

    return true;
}

void SdpaKernelEngine::getDetails(SdpaKernelHandle& /* handle*/,
                                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                                  hipdnnPluginConstData_t& /*detailsOut*/) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelEngine::getdetails not implemented");
}

size_t SdpaKernelEngine::getMaxWorkspaceSize(
    const SdpaKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelEngine::getMaxWorkspaceSize not implemented");
    return 0;
}

void SdpaKernelEngine::initializeExecutionContext(
    const SdpaKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    SdpaKernelContext& /*executionContext*/) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelEngine::initializeExecutionContext not implemented");
}

}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlanBuilder.hpp"
#include <iostream>

namespace sdpa_kernel_provider
{

bool SdpaKernelPlanBuilder::isApplicable(
    const SdpaKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto& nodeWrappers = opGraph.nodeWrappers();

    if(nodeWrappers.size() != 1
       || nodeWrappers.front()->attributesType()
              != hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes)
    {
        std::cout << "\n\n\n\nNodewrappers has incorrect size or wrong type\n\n\n\n\n";
        std::cout << "Nodewrappers.size() = " << nodeWrappers.size() << "\n";
        if(nodeWrappers.size() == 1)
        {
            std::cout << "front()->attributesType() = "
                      << static_cast<int>(nodeWrappers.front()->attributesType()) << "\n";
        }
        return false;
    }

    std::cout << "\n\n\n\nNodewrappers has correct type size or wrong type\n\n\n\n\n";

    // TODO: Add more expansive checks
    HIPDNN_PLUGIN_LOG_WARN("SdpaKernelPlanBuilder::isApplicable not fully implemented");

    return true;
}

size_t SdpaKernelPlanBuilder::getMaxWorkspaceSize(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const SdpaKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlanBuilder::getMaxWorkspaceSize not implemented");
    return 0;
}

void SdpaKernelPlanBuilder::initializeExecutionSettings(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlanBuilder::initializeExecutionContext not implemented");
}

void SdpaKernelPlanBuilder::buildPlan(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelContext& /* executionContext */) const
{
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaKernelPlanBuilder::getCustomKnobs(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace sdpa_kernel_provider

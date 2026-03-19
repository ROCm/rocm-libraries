// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlanBuilder.hpp"
#include "SdpaKernelPlan.hpp"
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
        return false;
    }

    // TODO: Add more expansive checks
    HIPDNN_PLUGIN_LOG_WARN("SdpaKernelPlanBuilder::isApplicable not fully implemented");

    return true;
}

size_t SdpaKernelPlanBuilder::getMaxWorkspaceSize(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const SdpaKernelSettings& /* executionSettings */) const
{
    // Forward-only kernel uses 64KB LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor, not workspace
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
    SdpaKernelContext& executionContext) const
{
    // Create plan (no workspace needed for forward-only POC)
    executionContext.setPlan(std::make_unique<SdpaKernelPlan>());
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaKernelPlanBuilder::getCustomKnobs(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace sdpa_kernel_provider

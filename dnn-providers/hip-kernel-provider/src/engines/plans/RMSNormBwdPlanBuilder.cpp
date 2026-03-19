// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "RMSNormPlanBuilder.hpp"
#include "engines/plans/RMSNormApplicabilityChecks.hpp"
#include "engines/plans/RMSNormBwdPlan.hpp"

namespace hip_kernel_provider
{

RMSNormPlanBuilder::RMSNormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                       const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool RMSNormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{

    switch(opGraph.nodeCount())
    {
    case 1:
    {
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is applicable only for single node graphs. "
                               "Graph has "
                               << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t RMSNormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    // RMSNorm plan builder does not require workspace size - TODO ??
    return 0u;
}

void RMSNormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings) const
{
}

namespace
{

void buildPlanSingleNode([[maybe_unused]] const HipKernelHandle& handle,
                         const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                         const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                         const IKernelCompiler& kernelCompiler,
                         const IDevicePropertyProvider& devicePropertyProvider,
                         HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::RMSNormBwdAttributes>();

    RMSNormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSNormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void RMSNormBwdPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipKernelContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building RMSNorm bwd inference plan for node: " << nodeName);
    buildPlanSingleNode(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> RMSNormBwdPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // hip_kernel_provider

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <exception>
#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <set>
#include <string>

#include "RMSnormBwdPlanBuilder.hpp"
#include "engines/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include "engines/plans/RMSnorm/RMSnormBwdPlan.hpp"

namespace hip_kernel_provider::rmsnorm
{

RMSnormBwdPlanBuilder::RMSnormBwdPlanBuilder(const IKernelCompiler& kernelCompiler,
                                             const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool RMSnormBwdPlanBuilder::isApplicable(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto anyNodeIsNotF32Compute = [&]() {
        return !std::all_of(
            opGraph.nodeWrappers().begin(), opGraph.nodeWrappers().end(), [](const auto& node) {
                return node->computeDataType() == hipdnn_data_sdk::data_objects::DataType::FLOAT;
            });
    };

    switch(opGraph.nodeCount())
    {
    case 1:
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "RMSnorm backward plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_data_sdk::data_objects::NodeAttributes>{
                   hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "RMSnorm backward plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkBwdTensorConfigSupported(
                *node.attributes_as_RMSNormBackwardAttributes());
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "RMSnorm backward plan builder is applicable only for single node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t RMSnormBwdPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    // RMSnorm backward plan currently does not require any workspace.
    return 0u;
}

void RMSnormBwdPlanBuilder::initializeExecutionSettings(
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
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes>();

    RMSnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void RMSnormBwdPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipKernelContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm backward plan for node: " << nodeName);
    buildPlanSingleNode(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> RMSnormBwdPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // hip_kernel_provider::rmsnorm

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "RMSnormApplicabilityChecks.hpp"
#include "RMSnormFwdPlan.hpp"
#include "RMSnormPlanBuilder.hpp"

#include <algorithm>
#include <set>

namespace hip_kernel_provider::rmsnorm
{

RMSnormPlanBuilder::RMSnormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                       const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool RMSnormPlanBuilder::isApplicable(
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
        // Kernel code always uses fp32 FLOAT_ACCUM
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR("RMSnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_data_sdk::data_objects::NodeAttributes>{
                   hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("RMSnorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkTensorConfigSupported(*node.attributes_as_RMSNormAttributes());
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
        HIPDNN_PLUGIN_LOG_INFO("RMSnorm plan builder is applicable only for single node graphs. "
                               "Graph has "
                               << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t RMSnormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    // RMS norm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanFwdSingleNode([[maybe_unused]] const HipKernelHandle& handle,
                            const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                            const IKernelCompiler& kernelCompiler,
                            const IDevicePropertyProvider& devicePropertyProvider,
                            HipKernelContext& executionContext)
{
    const auto& attr = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::RMSNormAttributes>();

    RMSnormFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormFwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void RMSnormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings) const
{
}

void RMSnormPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipKernelContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm fwd plan for node: " << nodeName);
    buildPlanFwdSingleNode(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> RMSnormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider

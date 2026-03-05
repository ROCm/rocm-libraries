// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "BatchnormPlanBuilder.hpp"
#include "engines/plans/BatchnormApplicabilityChecks.hpp"
#include "engines/plans/BatchnormFwdInferencePlan.hpp"

namespace hip_kernel_provider
{

BatchnormPlanBuilder::BatchnormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                           const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool BatchnormPlanBuilder::isApplicable(
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
            HIPDNN_PLUGIN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_data_sdk::data_objects::NodeAttributes>{
                   hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            checkBatchnormInferenceTensorConfigSupported(
                *node.attributes_as_BatchnormInferenceAttributes(), opGraph.getTensorMap());
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
        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is applicable only for single node graphs. "
                               "Graph has "
                               << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t BatchnormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void BatchnormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings) const
{
}

void BatchnormPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipKernelContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building batchnorm fwd inference plan for node: " << nodeName);
    buildPlanInferenceSingleNode(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> BatchnormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider

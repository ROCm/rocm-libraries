// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "BatchnormPlanBuilder.hpp"
#include "engines/plans/BatchnormApplicabilityChecks.hpp"
#include "engines/plans/BatchnormBwdPlan.hpp"
#include "engines/plans/BatchnormFwdInferencePlan.hpp"

namespace hip_kernel_provider
{

namespace
{

bool checkBwdSingleNodeApplicable(
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        const auto& bwdAttr
            = nodeWrapper
                  .attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

        if(!bwdAttr.mean_tensor_uid().has_value() || !bwdAttr.inv_variance_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO("Batchnorm backward requires saved mean and inv_variance for "
                                   "hip-kernel-provider");
            return false;
        }

        checkBatchnormBackwardTensorConfigSupported(bwdAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

bool checkBwdActivationFusionApplicable(
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& node0,
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& node1,
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& node2,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        const auto& bnInfAttr
            = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
        const auto& actAttr
            = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
        const auto& bnBwdAttr
            = node2.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

        if(!bnBwdAttr.mean_tensor_uid().has_value()
           || !bnBwdAttr.inv_variance_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO("Fused batchnorm backward requires saved mean and inv_variance");
            return false;
        }

        checkBatchnormInferenceActivationBackwardTensorConfigSupported(
            bnInfAttr, actAttr, bnBwdAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

} // namespace

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

        const auto& nodeWrapper = opGraph.getNodeWrapper(0);

        if(nodeWrapper.attributesType()
           == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes)
        {
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

        if(nodeWrapper.attributesType()
           == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes)
        {
            return checkBwdSingleNodeApplicable(nodeWrapper, opGraph.getTensorMap());
        }

        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
        return false;
    }
    case 3:
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        const auto& node0 = opGraph.getNodeWrapper(0);
        const auto& node1 = opGraph.getNodeWrapper(1);
        const auto& node2 = opGraph.getNodeWrapper(2);

        bool isBnInfFirst
            = node0.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
        bool isPointwiseSecond
            = node1.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes;
        bool isBnBwdThird
            = node2.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes;

        if(!(isBnInfFirst && isPointwiseSecond && isBnBwdThird))
        {
            HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder: 3-node graph must be "
                                   "BatchnormInference + Pointwise + BatchnormBackward");
            return false;
        }

        if(!checkBwdActivationFusionApplicable(node0, node1, node2, opGraph.getTensorMap()))
        {
            return false;
        }

        HIPDNN_PLUGIN_LOG_INFO(
            "Batchnorm plan builder applicable for batchnorm backward + activation fusion");
        return true;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is applicable only for 1 or 3 node graphs. "
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

void buildPlanBwdSingleNode([[maybe_unused]] const HipKernelHandle& handle,
                            const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                            HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBwdActivation([[maybe_unused]] const HipKernelHandle& handle,
                                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                                 HipKernelContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);
    const auto& node2 = opGraph.getNodeWrapper(2);

    const auto& bnInfAttr
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
    const auto& bnBwdAttr
        = node2.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(bnInfAttr, actAttr, bnBwdAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
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
    if(opGraph.nodeCount() == 3)
    {
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm backward + activation fusion plan");
        buildPlanFusedBwdActivation(handle, opGraph, executionContext);
        return;
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm fwd inference plan for node: " << nodeName);
        buildPlanInferenceSingleNode(handle,
                                     opGraph,
                                     nodeWrapper,
                                     _kernelCompiler,
                                     _devicePropertyProvider,
                                     executionContext);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm backward plan for node: " << nodeName);
        buildPlanBwdSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(
                    hipdnn_data_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> BatchnormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>
#include <unordered_set>

#include "BatchnormPlanBuilder.hpp"
#include "HipKernelUtils.hpp"
#include "engines/plans/BatchnormApplicabilityChecks.hpp"
#include "engines/plans/BatchnormBwdPlan.hpp"
#include "engines/plans/BatchnormFwdInferencePlan.hpp"

namespace hip_kernel_plugin
{

namespace
{
void batchnormFwdFusionCheckTensors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // in_0 must be the batchnorm inference output (forward path)
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = hip_kernel_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = hip_kernel_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}
bool batchnormFwdFusionCheckTensorsLogErrors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        batchnormFwdFusionCheckTensors(bnInfAttr, actAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

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

        if(!bwdAttr.mean_tensor_uid().has_value()
           || !bwdAttr.inv_variance_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Batchnorm backward requires saved mean and inv_variance for hip-kernel-provider");
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
            = node0
                  .attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
        const auto& actAttr
            = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
        const auto& bnBwdAttr
            = node2
                  .attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

        if(!bnBwdAttr.mean_tensor_uid().has_value()
           || !bnBwdAttr.inv_variance_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Fused batchnorm backward requires saved mean and inv_variance");
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

bool BatchnormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
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
    case 2:
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        const auto& node0 = opGraph.getNodeWrapper(0);
        const auto& node1 = opGraph.getNodeWrapper(1);

        bool isFwdInferenceFirst
            = node0.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
        bool isPointwiseSecond
            = node1.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes;

        if(!((isFwdInferenceFirst) && isPointwiseSecond))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Batchnorm plan builder is not applicable for this graph node order and types");
            return false;
        }

        if(isFwdInferenceFirst)
        {
            const auto& bnInfAttr
                = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
            const auto& actAttr
                = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
            if(!batchnormFwdFusionCheckTensorsLogErrors(bnInfAttr, actAttr, opGraph.getTensorMap()))
            {
                return false;
            }

            // Validate applicability before kernel dispatch by checking tensor configurations
            // and operation parameters manually.
            try
            {
                checkBatchnormInferenceActivationTensorConfigSupported(
                    bnInfAttr, actAttr, opGraph.getTensorMap());
            }
            catch(const std::exception& e)
            {
                HIPDNN_PLUGIN_LOG_INFO(e.what());
                return false;
            }
        }

        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                               "activation fusion");
        return true;
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
            HIPDNN_PLUGIN_LOG_INFO(
                "Batchnorm plan builder: 3-node graph must be "
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
        HIPDNN_PLUGIN_LOG_INFO(
            "Batchnorm plan builder is applicable only for 1, 2, or 3 node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t BatchnormPlanBuilder::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params),
                                                            executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedFwdInferenceActivation(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);

    const auto& fwdInference
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& activation
        = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();

    BatchnormFwdInferenceParams params(fwdInference, activation, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params),
                                                            executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan
        = std::make_unique<BatchnormBwdPlan>(std::move(params), executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBwdActivation(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);
    const auto& node2 = opGraph.getNodeWrapper(2);

    const auto& bnInfAttr
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr
        = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
    const auto& bnBwdAttr
        = node2.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(bnInfAttr, actAttr, bnBwdAttr, opGraph.getTensorMap());
    auto plan
        = std::make_unique<BatchnormBwdPlan>(std::move(params), executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void BatchnormPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() == 3)
    {
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm backward + activation fusion plan");
        buildPlanFusedBwdActivation(handle, opGraph, executionContext);
        return;
    }

    if(opGraph.nodeCount() == 2)
    {
        const auto& node0 = opGraph.getNodeWrapper(0);
        if(node0.attributesType()
           == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes)
        {
            HIPDNN_PLUGIN_LOG_INFO("Building batchnorm inference + activation fusion plan");
            buildPlanFusedFwdInferenceActivation(handle, opGraph, executionContext);
        }
        return;
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm fwd inference plan for node: " << nodeName);
        buildPlanInferenceSingleNode(handle, opGraph, nodeWrapper, executionContext);
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
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_plugin

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <miopen/miopen.h>
#include <string>

#include "MiopenBatchnormPlanBuilder.hpp"
#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

bool isNodeOfType(const hipdnn_plugin::IGraph& opGraph,
                  uint32_t index,
                  hipdnn_sdk::data_objects::NodeAttributes expectedType)
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(index);
    return nodeWrapper.isValid() && nodeWrapper.attributesType() == expectedType;
}

} // namespace

bool MiopenBatchnormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph) const
{
    switch(opGraph.nodeCount())
    {
    case 1:
    {
        if(!opGraph.hasOnlySupportedAttributes(std::set<hipdnn_sdk::data_objects::NodeAttributes>{
               hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
               hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes}))
        {
            HIPDNN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
            return false;
        }
        return true;
    }
    case 3:
    {
        // batchnorm inference -> activation -> batchnorm backward
        auto isBatchnormInferenceFirst = isNodeOfType(
            opGraph, 0, hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
        auto isActivationSecond = isNodeOfType(
            opGraph, 1, hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
        auto isBatchnormBwdThird = isNodeOfType(
            opGraph, 2, hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);

        auto isCorrectOrder
            = isBatchnormInferenceFirst && isActivationSecond && isBatchnormBwdThird;
        if(isCorrectOrder)
        {
            HIPDNN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                            "activation + batchnorm backward fusion");
            return true;
        }

        HIPDNN_LOG_INFO("Batchnorm plan builder requires batchnorm inference, activation, "
                        "batchnorm backward order. Current order not supported");
        return false;
    }
    default:
    {
        HIPDNN_LOG_INFO(
            "Batchnorm plan builder is applicable only for 1 or 3 node graphs. Graph has {} nodes",
            opGraph.nodeCount());
        return false;
    }
    }
}

size_t MiopenBatchnormPlanBuilder::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_plugin::IGraph& opGraph) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                  const hipdnn_plugin::IGraph& opGraph,
                                  const hipdnn_plugin::INodeWrapper& nodeWrapper,
                                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::IGraph& opGraph,
                            const hipdnn_plugin::INodeWrapper& nodeWrapper,
                            HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBackwardsActivation([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                       const hipdnn_plugin::IGraph& opGraph,
                                       HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);
    const auto& node2 = opGraph.getNodeWrapper(2);

    const auto& inferenceAttr
        = node0.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr = node1.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();
    const auto& bnAttr
        = node2.attributesAs<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(bnAttr, actAttr, inferenceAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace

void MiopenBatchnormPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() == 3)
    {
        HIPDNN_LOG_INFO(
            "Building batchnorm inference + activation + batchnorm backward fusion plan");
        buildPlanFusedBackwardsActivation(handle, opGraph, executionContext);
        return;
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto& node = nodeWrapper.node();
    const auto nodeName = node.name() != nullptr ? node.name()->str() : "";

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", nodeName);
        buildPlanInferenceSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_LOG_INFO("Building batchnorm backward plan for node: {}", nodeName);
        buildPlanBwdSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

} // namespace miopen_legacy_plugin

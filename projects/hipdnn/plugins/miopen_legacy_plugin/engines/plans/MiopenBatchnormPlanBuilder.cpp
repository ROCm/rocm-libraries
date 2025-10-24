/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

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

bool MiopenBatchnormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() == 1)
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

    if(opGraph.nodeCount() == 3)
    {
        const auto& node0 = opGraph.getNode(0);
        const auto& node1 = opGraph.getNode(1);
        const auto& node2 = opGraph.getNode(2);

        // batchnorm inference -> activation -> batchnorm backward
        bool isBatchnormInferenceFirst
            = (node0.attributes_type()
               == hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
        bool isActivationSecond
            = (node1.attributes_type()
               == hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
        bool isBatchnormBwdThird
            = (node2.attributes_type()
               == hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);

        bool isCorrectOrder
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

    HIPDNN_LOG_INFO(
        "Batchnorm plan builder is applicable only for 1 or 3 node graphs. Graph has {} nodes",
        opGraph.nodeCount());
    return false;
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

std::string getNodeName(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

void buildPlanInferenceSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                  const hipdnn_plugin::IGraph& opGraph,
                                  const hipdnn_sdk::data_objects::Node& node,
                                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto* attr = node.attributes_as_BatchnormInferenceAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormInferenceAttributes for node: "
                + getNodeName(node));
    }

    BatchnormFwdInferenceParams params(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::IGraph& opGraph,
                            const hipdnn_sdk::data_objects::Node& node,
                            HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto* attr = node.attributes_as_BatchnormBackwardAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormBackwardAttributes for node: "
                + getNodeName(node));
    }

    BatchnormBwdParams params(*attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBackwardsActivation([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                       const hipdnn_plugin::IGraph& opGraph,
                                       HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node0 = opGraph.getNode(0);
    const auto& node1 = opGraph.getNode(1);
    const auto& node2 = opGraph.getNode(2);

    const auto* inferenceAttr = node0.attributes_as_BatchnormInferenceAttributes();
    if(inferenceAttr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert attributes to BatchnormInferenceAttributes for node: "
                + getNodeName(node0));
    }

    const auto* actAttr = node1.attributes_as_PointwiseAttributes();
    if(actAttr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert attributes to PointwiseAttributes for node: " + getNodeName(node1));
    }

    const auto* bnAttr = node2.attributes_as_BatchnormBackwardAttributes();
    if(bnAttr == nullptr)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert attributes to BatchnormBackwardAttributes for node: "
                + getNodeName(node2));
    }

    BatchnormBwdParams params(*bnAttr, *actAttr, *inferenceAttr, opGraph.getTensorMap());
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

    const auto& node = opGraph.getNode(0);
    std::string nodeName = getNodeName(node);
    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", nodeName);
        buildPlanInferenceSingleNode(handle, opGraph, node, executionContext);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_LOG_INFO("Building batchnorm backward plan for node: {}", nodeName);
        buildPlanBwdSingleNode(handle, opGraph, node, executionContext);
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

} // namespace miopen_legacy_plugin

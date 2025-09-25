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
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_LOG_INFO(
            "Batchnorm plan builder is applicable only for single node graphs. Graph has {} nodes",
            opGraph.nodeCount());
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(std::set<hipdnn_sdk::data_objects::NodeAttributes>{
           hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
           hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes}))
    {
        HIPDNN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
        return false;
    }

    return true;
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

} // namespace

void MiopenBatchnormPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <miopen/miopen.h>
#include <string>

#include "MiopenBatchnormPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

std::tuple<const hipdnn_sdk::data_objects::BatchnormInferenceAttributes&,
           const hipdnn_sdk::data_objects::PointwiseAttributes&,
           const hipdnn_sdk::data_objects::BatchnormBackwardAttributes&>
    getBatchnormBackwardFusionNodeAttrs(const hipdnn_plugin::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 3)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fusion requires exactly 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& bnInfAttr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();

    const auto& actAttr
        = opGraph.getNodeWrapper(1).attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

    const auto& bnBwdAttr
        = opGraph.getNodeWrapper(2)
              .attributesAs<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>();

    return {bnInfAttr, actAttr, bnBwdAttr};
}

auto getBatchnormBackwardFusionNodeAttrsNoExcept(const hipdnn_plugin::IGraph& opGraph)
    -> std::optional<decltype(getBatchnormBackwardFusionNodeAttrs(opGraph))>
{
    try
    {
        return getBatchnormBackwardFusionNodeAttrs(opGraph);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return {};
    }
}

void batchnormFusionCheckTensors(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    // Verify inference output is activation input
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid()
       && (!actAttr.in_1_tensor_uid().has_value()
           || actAttr.in_1_tensor_uid().value() != bnInfAttr.y_tensor_uid()))
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation node input must be the batchnorm inference output tensor");
    }

    // Verify activation backwards output is BN backward dy input
    if(actAttr.out_0_tensor_uid() != bnBwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be the activation output tensor");
    }

    // Verify that different BN operations use shared inputs where applicable
    if(bnBwdAttr.x_tensor_uid() != bnInfAttr.x_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same X tensor as batchnorm inference");
    }

    if(bnBwdAttr.mean_tensor_uid().has_value()
       && bnBwdAttr.mean_tensor_uid().value() != bnInfAttr.mean_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same mean tensor as batchnorm inference");
    }

    if(bnBwdAttr.inv_variance_tensor_uid().has_value()
       && bnBwdAttr.inv_variance_tensor_uid().value() != bnInfAttr.inv_variance_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same inv_variance tensor as batchnorm inference");
    }

    if(bnBwdAttr.scale_tensor_uid() != bnInfAttr.scale_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same scale tensor as batchnorm inference");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    const auto& actBnInputTensor
        = (actAttr.in_0_tensor_uid() == bnInfAttr.y_tensor_uid())
              ? actTensorIn0
              : miopen_utils::findTensorAttributes(tensorMap, actAttr.in_1_tensor_uid().value());

    if(!actBnInputTensor.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be virtual");
    }

    const auto& bnBwdTensorDy
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dy_tensor_uid());
    const auto& bnBwdTensorDx
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dx_tensor_uid());
    const auto& bnBwdTensorDscale
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dscale_tensor_uid());
    const auto& bnBwdTensorDbias
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dbias_tensor_uid());

    if(!bnBwdTensorDy.virtual_() || bnBwdTensorDx.virtual_() || bnBwdTensorDscale.virtual_()
       || bnBwdTensorDbias.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

bool batchnormFusionCheckTensorsNoExcept(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    try
    {
        batchnormFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, tensorMap);
        return true;
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }
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
        const auto nodeAttrs = getBatchnormBackwardFusionNodeAttrsNoExcept(opGraph);
        if(!nodeAttrs.has_value())
        {
            return false;
        }

        if(!batchnormFusionCheckTensorsNoExcept(std::get<0>(nodeAttrs.value()),
                                                std::get<1>(nodeAttrs.value()),
                                                std::get<2>(nodeAttrs.value()),
                                                opGraph.getTensorMap()))
        {
            return false;
        }

        HIPDNN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                        "activation + batchnorm backward fusion");
        return true;
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
    const auto [bnInfAttr, actAttr, bnBwdAttr] = getBatchnormBackwardFusionNodeAttrs(opGraph);
    batchnormFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, opGraph.getTensorMap());

    BatchnormBwdParams params(bnBwdAttr, actAttr, bnInfAttr, opGraph.getTensorMap());
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
    const auto nodeName = nodeWrapper.name();

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

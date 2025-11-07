// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>
#include <tuple>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenBatchnormFwdTrainingActivPlanBuilder.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingActivPlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

bool isNodeActivFwd(const hipdnn_sdk::data_objects::PointwiseAttributes& attr)
{
    using PointwiseMode = hipdnn_sdk::data_objects::PointwiseMode;

    if(attr.operation() != PointwiseMode::RELU_FWD)
    {
        return false;
    }

    // MIOpen batchnorm fusion supports:
    // - Standard ReLU (no parameters)
    // - Clipped ReLU (relu_upper_clip only)
    // - CLAMP (relu_lower_clip + relu_upper_clip)
    // But does NOT support Leaky ReLU (relu_lower_clip_slope)
    if(attr.relu_lower_clip_slope())
    {
        return false; // Leaky ReLU not supported
    }

    return true;
}

std::tuple<const hipdnn_sdk::data_objects::BatchnormAttributes&,
           const hipdnn_sdk::data_objects::PointwiseAttributes&>
    getNodeAttrs(const hipdnn_plugin::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 2)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "BatchnormFwdTrainingActiv  plan builder supports only graphs with 2 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    // Expect that the graph is sorted in topological order
    // Expect the first node to be batchnorm forward training operation
    const auto& bnNodeWrapper = opGraph.getNodeWrapper(0);
    const auto bnNodeName = bnNodeWrapper.name();
    if(bnNodeWrapper.attributesType()
       != hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "First node in the graph (" + bnNodeName
                + ") must be batchnorm forward training. Found node of type: "
                + std::string(hipdnn_sdk::data_objects::toString(bnNodeWrapper.attributesType())));
    }
    const auto& bnAttr
        = bnNodeWrapper.attributesAs<hipdnn_sdk::data_objects::BatchnormAttributes>();

    // Expect the second node to be activation forward
    const auto& activNodeWrapper = opGraph.getNodeWrapper(1);
    const auto activNodeName = activNodeWrapper.name();
    if(activNodeWrapper.attributesType()
       != hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Second node in the graph (" + activNodeName
                + ") must be pointwise operation. Found node of type: "
                + std::string(
                    hipdnn_sdk::data_objects::toString(activNodeWrapper.attributesType())));
    }
    const auto& activAttr
        = activNodeWrapper.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

    if(!isNodeActivFwd(activAttr))
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Second node in the graph (" + activNodeName
                + ") must be activation forward. Found pointwise operation: "
                + std::string(hipdnn_sdk::data_objects::toString(activAttr.operation())));
    }

    return {bnAttr, activAttr};
}

auto getNodeAttrsLogErrors(const hipdnn_plugin::IGraph& opGraph)
    -> std::optional<decltype(getNodeAttrs(opGraph))>
{
    try
    {
        return getNodeAttrs(opGraph);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return {};
    }
}

void nodeAttrsCheckTensors(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    // Check the connection between batchnorm and activation nodes
    // The activation input must be the batchnorm output
    if(activAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation node input must be the batchnorm node output tensor");
    }

    // Verify that all tensors are either virtual or non-virtual as expected
    // Batchnorm: x, scale, bias must be non-virtual, y must be virtual
    const auto& bnTensorAttrX
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorAttrScale
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorAttrBias
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorAttrY
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());

    if(bnTensorAttrX.virtual_() || bnTensorAttrScale.virtual_() || bnTensorAttrBias.virtual_()
       || !bnTensorAttrY.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm x, scale, bias tensors must be non-virtual, y tensor must be virtual");
    }

    // Saved batch statistics (mean/invVariance) must be non-virtual if present
    if(bnAttr.mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrMean
            = miopen_utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value());
        if(bnTensorAttrMean.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorAttrInvVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorAttrInvVar.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

    // Optional running statistics tensors must be non-virtual if present
    if(bnAttr.prev_running_mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrPrevRunningMean = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.prev_running_mean_tensor_uid().value());
        if(bnTensorAttrPrevRunningMean.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm prev_running_mean tensor must be non-virtual");
        }
    }

    if(bnAttr.prev_running_variance_tensor_uid().has_value())
    {
        const auto& bnTensorAttrPrevRunningVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.prev_running_variance_tensor_uid().value());
        if(bnTensorAttrPrevRunningVar.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm prev_running_variance tensor must be non-virtual");
        }
    }

    if(bnAttr.next_running_mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrNextRunningMean = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.next_running_mean_tensor_uid().value());
        if(bnTensorAttrNextRunningMean.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm next_running_mean tensor must be non-virtual");
        }
    }

    if(bnAttr.next_running_variance_tensor_uid().has_value())
    {
        const auto& bnTensorAttrNextRunningVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.next_running_variance_tensor_uid().value());
        if(bnTensorAttrNextRunningVar.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm next_running_variance tensor must be non-virtual");
        }
    }

    // Activation: input from batchnorm must be virtual, output must be non-virtual
    const auto& activInAttr
        = miopen_utils::findTensorAttributes(tensorMap, activAttr.in_0_tensor_uid());
    const auto& activOutAttr
        = miopen_utils::findTensorAttributes(tensorMap, activAttr.out_0_tensor_uid());

    if(!activInAttr.virtual_() || activOutAttr.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation node input must be virtual, output must be non-virtual");
    }
}

bool nodeAttrsCheckTensorsLogErrors(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    try
    {
        nodeAttrsCheckTensors(bnAttr, activAttr, tensorMap);
        return true;
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }
}

} // namespace

bool MiopenBatchnormFwdTrainingActivPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph) const
{
    const auto nodeAttrs = getNodeAttrsLogErrors(opGraph);
    if(!nodeAttrs.has_value())
    {
        return false;
    }

    const auto& bnAttr = std::get<0>(nodeAttrs.value());

    // NOTE: Running statistics support is disabled pending MIOpen API update.
    // Remove this block when MIOpen supports separate input/output buffers for running statistics.
    // hipDNN's graph API uses separate prev_running_mean/variance (input) and next_running_mean/variance (output)
    // buffers, but MIOpen's API requires single IN/OUT buffers for running statistics.
    // This cannot be correctly bridged without either:
    // 1. Updating MIOpen API to support separate input/output buffers, or
    // 2. Implementing buffer copy operations (with performance overhead)
#if 0 // TODO: Change to 1 when MIOpen API is updated to support separate input/output buffers
    // Running statistics will be supported - no rejection needed
#else
    // Until MIOpen is updated, reject graphs with running statistics
    if(bnAttr.prev_running_mean_tensor_uid().has_value()
       || bnAttr.prev_running_variance_tensor_uid().has_value()
       || bnAttr.momentum_tensor_uid().has_value()
       || bnAttr.next_running_mean_tensor_uid().has_value()
       || bnAttr.next_running_variance_tensor_uid().has_value())
    {
        HIPDNN_LOG_INFO("Running statistics not yet supported - MIOpen API update required");
        return false;
    }
#endif

    if(!nodeAttrsCheckTensorsLogErrors(
           bnAttr, std::get<1>(nodeAttrs.value()), opGraph.getTensorMap()))
    {
        return false;
    }

    try
    {
        // Validate params can be created successfully
        BatchnormFwdTrainingActivParams params(
            bnAttr, std::get<1>(nodeAttrs.value()), opGraph.getTensorMap());
        return true;
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        return false;
    }
}

size_t MiopenBatchnormFwdTrainingActivPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph) const
{
    const auto [bnAttr, activAttr] = getNodeAttrs(opGraph);
    nodeAttrsCheckTensors(bnAttr, activAttr, opGraph.getTensorMap());

    BatchnormFwdTrainingActivParams params(bnAttr, activAttr, opGraph.getTensorMap());
    BatchnormFwdTrainingActivPlan plan(handle, std::move(params));
    return plan.getWorkspaceSize(handle);
}

void MiopenBatchnormFwdTrainingActivPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    const auto [bnAttr, activAttr] = getNodeAttrs(opGraph);
    nodeAttrsCheckTensors(bnAttr, activAttr, opGraph.getTensorMap());

    BatchnormFwdTrainingActivParams params(bnAttr, activAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdTrainingActivPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace miopen_legacy_plugin

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>
#include <unordered_set>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenBatchnormFwdTrainingPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

bool isNodeActivFwd(const hipdnn_sdk::data_objects::PointwiseAttributes& attr)
{
    using PM = hipdnn_sdk::data_objects::PointwiseMode;
    static const std::unordered_set<PM> s_supportedActivations = {PM::RELU_FWD, PM::IDENTITY};

    if(s_supportedActivations.count(attr.operation()) == 0)
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

void checkTensorVirtuality1Node(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    // Check for virtual tensors - 1-node case (solo batchnorm training)
    const auto& bnTensorX = miopen_utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorScale
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorBias
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorY = miopen_utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());

    if(bnTensorX.virtual_() || bnTensorScale.virtual_() || bnTensorBias.virtual_()
       || bnTensorY.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm training tensors must be non-virtual for 1-node graph");
    }

    // Optional mean/variance tensors must be non-virtual if present
    if(bnAttr.mean_tensor_uid().has_value())
    {
        const auto& bnTensorMean
            = miopen_utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value());
        if(bnTensorMean.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

#if 0 // TODO: Enable running statistics validation when MIOpen API supports separate input/output buffers
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
#endif
}

void checkTensorVirtuality2Node(
    const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    // Check for virtual tensors - 2-node case (batchnorm training + activation)
    const auto& bnTensorX = miopen_utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorScale
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorBias
        = miopen_utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorY = miopen_utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());

    if(bnTensorX.virtual_() || bnTensorScale.virtual_() || bnTensorBias.virtual_()
       || !bnTensorY.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm training input tensors must be non-virtual, output tensor must be virtual");
    }

    // Optional mean/variance tensors must be non-virtual if present
    if(bnAttr.mean_tensor_uid().has_value())
    {
        const auto& bnTensorMean
            = miopen_utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value());
        if(bnTensorMean.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

#if 0 // TODO: Enable running statistics validation when MIOpen API supports separate input/output buffers
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
#endif

    const auto& actTensorIn0
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

} // namespace

bool MiopenBatchnormFwdTrainingPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() == 1)
    {
        // Solo batchnorm training
        const auto& node = opGraph.getNodeWrapper(0);
        if(node.attributesType() != hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes)
        {
            return false;
        }

        const auto& bnAttr = node.attributesAs<hipdnn_sdk::data_objects::BatchnormAttributes>();

#if 0 // TODO: Remove this block when MIOpen API supports separate input/output buffers for running statistics \
    // Running statistics will be supported - no rejection needed
#else
        // Until MIOpen is updated, reject graphs with running statistics
        // API mismatch: hipDNN graph API uses separate prev/next buffers for running statistics,
        // but MIOpen requires single IN/OUT buffers. This cannot be correctly bridged without
        // either updating MIOpen API or implementing buffer copy operations.
        if(bnAttr.prev_running_mean_tensor_uid().has_value()
           || bnAttr.prev_running_variance_tensor_uid().has_value()
           || bnAttr.momentum_tensor_uid().has_value()
           || bnAttr.next_running_mean_tensor_uid().has_value()
           || bnAttr.next_running_variance_tensor_uid().has_value())
        {
            HIPDNN_LOG_INFO("Batchnorm fwd training plan builder does not support running "
                            "statistics - MIOpen API update required");
            return false;
        }
#endif

        try
        {
            checkTensorVirtuality1Node(bnAttr, opGraph.getTensorMap());
            return true;
        }
        catch(const hipdnn_plugin::HipdnnPluginException& e)
        {
            HIPDNN_LOG_INFO(e.what());
            return false;
        }
    }
    else if(opGraph.nodeCount() == 2)
    {
        // Batchnorm training + activation fusion
        const auto& node0 = opGraph.getNodeWrapper(0);
        const auto& node1 = opGraph.getNodeWrapper(1);

        if(node0.attributesType() != hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes
           || node1.attributesType()
                  != hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes)
        {
            return false;
        }

        const auto& bnAttr = node0.attributesAs<hipdnn_sdk::data_objects::BatchnormAttributes>();
        const auto& activAttr = node1.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

        // Check if activation is supported
        if(!isNodeActivFwd(activAttr))
        {
            HIPDNN_LOG_INFO("Unsupported activation mode for batchnorm fusion");
            return false;
        }

        // Validate that activation input matches batchnorm output
        if(activAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
        {
            HIPDNN_LOG_INFO("Activation input must match batchnorm output");
            return false;
        }

#if 0 // TODO: Remove this block when MIOpen API supports separate input/output buffers for running statistics \
    // Running statistics will be supported - no rejection needed
#else
        // Until MIOpen is updated, reject graphs with running statistics
        // API mismatch: hipDNN graph API uses separate prev/next buffers for running statistics,
        // but MIOpen requires single IN/OUT buffers. This cannot be correctly bridged without
        // either updating MIOpen API or implementing buffer copy operations.
        if(bnAttr.prev_running_mean_tensor_uid().has_value()
           || bnAttr.prev_running_variance_tensor_uid().has_value()
           || bnAttr.momentum_tensor_uid().has_value()
           || bnAttr.next_running_mean_tensor_uid().has_value()
           || bnAttr.next_running_variance_tensor_uid().has_value())
        {
            HIPDNN_LOG_INFO("Batchnorm fwd training plan builder does not support running "
                            "statistics - MIOpen API update required");
            return false;
        }
#endif

        try
        {
            checkTensorVirtuality2Node(bnAttr, activAttr, opGraph.getTensorMap());
            // Validate params can be created successfully
            BatchnormFwdTrainingParams params(bnAttr, activAttr, opGraph.getTensorMap());
            return true;
        }
        catch(const hipdnn_plugin::HipdnnPluginException& e)
        {
            HIPDNN_LOG_INFO(e.what());
            return false;
        }
    }

    return false;
}

size_t MiopenBatchnormFwdTrainingPlanBuilder::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_plugin::IGraph& opGraph) const
{
    // No workspace needed for batchnorm forward training
    return 0;
}

void MiopenBatchnormFwdTrainingPlanBuilder::buildPlan(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() == 1)
    {
        // Solo batchnorm training
        const auto& bnAttr = opGraph.getNodeWrapper(0)
                                 .attributesAs<hipdnn_sdk::data_objects::BatchnormAttributes>();

        checkTensorVirtuality1Node(bnAttr, opGraph.getTensorMap());

        BatchnormFwdTrainingParams params(bnAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
        executionContext.setPlan(std::move(plan));
    }
    else if(opGraph.nodeCount() == 2)
    {
        // Batchnorm training + activation fusion
        const auto& bnAttr = opGraph.getNodeWrapper(0)
                                 .attributesAs<hipdnn_sdk::data_objects::BatchnormAttributes>();
        const auto& activAttr = opGraph.getNodeWrapper(1)
                                    .attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

        checkTensorVirtuality2Node(bnAttr, activAttr, opGraph.getTensorMap());

        BatchnormFwdTrainingParams params(bnAttr, activAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
        executionContext.setPlan(std::move(plan));
    }
    else
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fwd training plan builder supports only 1 or 2 node graphs");
    }
}

} // namespace miopen_legacy_plugin

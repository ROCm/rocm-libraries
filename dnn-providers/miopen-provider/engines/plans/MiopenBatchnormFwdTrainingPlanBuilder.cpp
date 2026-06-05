// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenBatchnormFwdTrainingPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBatchnormApplicabilityChecks.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"

namespace miopen_plugin
{

namespace
{

bool isNodeActivFwd(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    // Check if operation is supported for batchnorm fusion
    switch(attr.operation())
    {
    case PointwiseMode::RELU_FWD:
        break; // Continue to parameter validation
    default:
        return false;
    }

    // MIOpen batchnorm fusion supports:
    // - Standard ReLU (no parameters)
    // - Clipped ReLU (relu_upper_clip only)
    // - CLAMP (relu_lower_clip + relu_upper_clip)
    // But does NOT support Leaky ReLU (relu_lower_clip_slope)
    return !attr.relu_lower_clip_slope();
}

const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes&
    checkBatchnormNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node)
{
    if(node.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "First node must be batchnorm");
    }

    const auto& bnAttr
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();

    return bnAttr;
}

const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes&
    checkActivationNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
                        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr)
{
    if(node.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Second node must be pointwise");
    }

    const auto& activAttr
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    if(!isNodeActivFwd(activAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported activation mode for batchnorm fusion");
    }

    if(activAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation input must match batchnorm output");
    }

    return activAttr;
}

void checkRunningStatisticsTensorVirtuality(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Optional running statistics tensors must be non-virtual if present
    if(bnAttr.prev_running_mean_tensor_uid().has_value())
    {
        const auto& bnTensorAttrPrevRunningMean = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.prev_running_mean_tensor_uid().value());
        if(bnTensorAttrPrevRunningMean.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
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
            throw hipdnn_plugin_sdk::HipdnnPluginException(
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
            throw hipdnn_plugin_sdk::HipdnnPluginException(
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
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm next_running_variance tensor must be non-virtual");
        }
    }
}

void checkTensorVirtuality1Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
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
        throw hipdnn_plugin_sdk::HipdnnPluginException(
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
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

    checkRunningStatisticsTensorVirtuality(bnAttr, tensorMap);
}

void checkTensorVirtuality2Node(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& bnAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
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
        throw hipdnn_plugin_sdk::HipdnnPluginException(
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
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm mean tensor must be non-virtual");
        }
    }

    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        const auto& bnTensorInvVar = miopen_utils::findTensorAttributes(
            tensorMap, bnAttr.inv_variance_tensor_uid().value());
        if(bnTensorInvVar.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm inv_variance tensor must be non-virtual");
        }
    }

    checkRunningStatisticsTensorVirtuality(bnAttr, tensorMap);

    const auto& actTensorIn0
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

} // namespace

bool MiopenBatchnormFwdTrainingPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1 && opGraph.nodeCount() != 2)
    {
        return false;
    }

    try
    {
        // Common batchnorm validation
        const auto& bnAttr = checkBatchnormNode(opGraph.getNodeWrapper(0));

        auto hasFloatComputeDataType = [](const auto& node) {
            return node->computeDataType() == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        };

        if(!std::all_of(opGraph.nodeWrappers().begin(),
                        opGraph.nodeWrappers().end(),
                        hasFloatComputeDataType))
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "BatchnormFwdTraining plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        if(opGraph.nodeCount() == 1)
        {
            // Solo batchnorm training
            checkTensorVirtuality1Node(bnAttr, opGraph.getTensorMap());

            // Since MIOpen does not provide an API to validate batchnorm applicability, we perform the
            // checks manually.
            checkBatchnormFwdTrainingTensorConfigSupported(bnAttr, opGraph.getTensorMap());

            HIPDNN_PLUGIN_LOG_INFO("BatchnormFwdTraining plan builder applicable for single node "
                                   "batchnorm training");
            return true;
        }

        // nodeCount == 2: Batchnorm training + activation fusion
        const auto& activAttr = checkActivationNode(opGraph.getNodeWrapper(1), bnAttr);
        checkTensorVirtuality2Node(bnAttr, activAttr, opGraph.getTensorMap());

        checkBatchnormFwdTrainingActivationTensorConfigSupported(
            bnAttr, activAttr, opGraph.getTensorMap());
        checkBatchnormFwdActivationModeSupported(activAttr);

        HIPDNN_PLUGIN_LOG_INFO(
            "BatchnormFwdTraining plan builder applicable for training + activation fusion");
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

size_t MiopenBatchnormFwdTrainingPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const
{
    // No workspace needed for batchnorm forward training
    return 0;
}

void MiopenBatchnormFwdTrainingPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
}

void MiopenBatchnormFwdTrainingPlanBuilder::buildPlan(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    if(opGraph.nodeCount() == 1)
    {
        // Solo batchnorm training
        const auto& bnAttr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();

        BatchnormFwdTrainingParams params(bnAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(
            std::move(params), executionContext.executionSettings());
        executionContext.setPlan(std::move(plan));
    }
    else if(opGraph.nodeCount() == 2)
    {
        // Batchnorm training + activation fusion
        const auto& bnAttr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();
        const auto& activAttr
            = opGraph.getNodeWrapper(1)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

        BatchnormFwdTrainingParams params(bnAttr, activAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(
            std::move(params), executionContext.executionSettings());
        executionContext.setPlan(std::move(plan));
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fwd training plan builder supports only 1 or 2 node graphs");
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    MiopenBatchnormFwdTrainingPlanBuilder::getCustomKnobs(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace miopen_plugin

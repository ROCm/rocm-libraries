// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include "BatchnormPlanBuilder.hpp"
#include "engines/plans/batchnorm/BatchnormApplicabilityChecks.hpp"
#include "engines/plans/batchnorm/BatchnormBwdPlan.hpp"
#include "engines/plans/batchnorm/BatchnormFwdInferencePlan.hpp"
#include "engines/plans/batchnorm/BatchnormFwdInferenceWithVariancePlan.hpp"
#include "engines/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"

namespace hip_kernel_provider::batchnorm
{

namespace
{
std::tuple<const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes&,
           const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes&,
           const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes&>
    getBatchnormBackwardFusionNodeAttrs(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fusion requires exactly 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& bnInfAttr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr
        = opGraph.getNodeWrapper(1)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();
    const auto& bnBwdAttr
        = opGraph.getNodeWrapper(2)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes>();

    return {bnInfAttr, actAttr, bnBwdAttr};
}

auto getBatchnormBackwardFusionNodeAttrsLogErrors(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
    -> std::optional<decltype(getBatchnormBackwardFusionNodeAttrs(opGraph))>
{
    try
    {
        return getBatchnormBackwardFusionNodeAttrs(opGraph);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return {};
    }
}

void batchnormBwdFusionCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    if(!actAttr.in_1_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward requires in_1 tensor (dy gradient)");
    }

    if(actAttr.out_0_tensor_uid() != bnBwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be the activation output tensor");
    }

    if(bnBwdAttr.x_tensor_uid() != bnInfAttr.x_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same X tensor as batchnorm inference");
    }

    if(bnBwdAttr.mean_tensor_uid().has_value()
       && bnBwdAttr.mean_tensor_uid().value() != bnInfAttr.mean_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same mean tensor as batchnorm inference");
    }

    if(bnBwdAttr.inv_variance_tensor_uid().has_value()
       && bnBwdAttr.inv_variance_tensor_uid().value() != bnInfAttr.inv_variance_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same inv_variance tensor as batchnorm inference");
    }

    if(bnBwdAttr.scale_tensor_uid() != bnInfAttr.scale_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same scale tensor as batchnorm inference");
    }

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

    const auto& actTensorIn1
        = hip_kernel_utils::findTensorAttributes(tensorMap, actAttr.in_1_tensor_uid().value());
    if(actTensorIn1.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation in_1 (dy gradient) must be non-virtual");
    }

    const auto& actTensorIn0
        = hip_kernel_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = hip_kernel_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn0.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be virtual");
    }

    const auto& bnBwdTensorDy
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnBwdAttr.dy_tensor_uid());
    const auto& bnBwdTensorDx
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnBwdAttr.dx_tensor_uid());
    const auto& bnBwdTensorDscale
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnBwdAttr.dscale_tensor_uid());
    const auto& bnBwdTensorDbias
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnBwdAttr.dbias_tensor_uid());

    if(!bnBwdTensorDy.virtual_() || bnBwdTensorDx.virtual_() || bnBwdTensorDscale.virtual_()
       || bnBwdTensorDbias.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

bool batchnormBwdFusionCheckTensorsLogErrors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        batchnormBwdFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

void batchnormFwdFusionCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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

void batchnormFwdWithVarianceFusionCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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
    const auto& bnInfTensorVariance
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.variance_tensor_uid());
    const auto& bnInfTensorScale
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = hip_kernel_utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorVariance.virtual_()
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

bool batchnormFwdWithVarianceFusionCheckTensorsLogErrors(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        batchnormFwdWithVarianceFusionCheckTensors(bnInfAttr, actAttr, tensorMap);
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
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto anyNodeIsNotF32Compute = [&]() {
        return !std::all_of(
            opGraph.nodeWrappers().begin(), opGraph.nodeWrappers().end(), [](const auto& node) {
                return node->computeDataType()
                       == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
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

        if(!opGraph.hasOnlySupportedAttributes(std::set<hipdnn_flatbuffers_sdk::data_objects::
                                                            NodeAttributes>{
               hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
               hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
               hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                   BatchnormInferenceAttributesVarianceExt,
               hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            BatchnormValidator validator(opGraph.getTensorMap());
            switch(node.attributes_type())
            {
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes:
                validator.checkFwdTrainingTensorConfigSupported(
                    *node.attributes_as_BatchnormAttributes());
                break;
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
                validator.checkInferenceTensorConfigSupported(
                    *node.attributes_as_BatchnormInferenceAttributes());
                break;
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                BatchnormInferenceAttributesVarianceExt:
                validator.checkInferenceVarianceExtTensorConfigSupported(
                    *node.attributes_as_BatchnormInferenceAttributesVarianceExt());
                break;
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
                validator.checkBwdTensorConfigSupported(
                    *node.attributes_as_BatchnormBackwardAttributes());
                break;
            default:
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "Unexpected node attribute type");
            }
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
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
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
        bool isFwdInferenceWithVarianceFirst
            = node0.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                  BatchnormInferenceAttributesVarianceExt;
        bool isPointwiseSecond
            = node1.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;

        if(!((isFwdInferenceFirst || isFwdInferenceWithVarianceFirst) && isPointwiseSecond))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Batchnorm plan builder is not applicable for this graph node order and types");
            return false;
        }

        BatchnormValidator validator(opGraph.getTensorMap());
        if(isFwdInferenceFirst)
        {
            const auto& bnInfAttr = node0.attributesAs<
                hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>();
            const auto& actAttr
                = node1.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();
            if(!batchnormFwdFusionCheckTensorsLogErrors(bnInfAttr, actAttr, opGraph.getTensorMap()))
            {
                return false;
            }

            try
            {
                validator.checkInferenceActivationTensorConfigSupported(bnInfAttr, actAttr);
            }
            catch(const std::exception& e)
            {
                HIPDNN_PLUGIN_LOG_INFO(e.what());
                return false;
            }
        }
        else
        {
            const auto& bnInfAttr = node0.attributesAs<
                hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();
            const auto& actAttr
                = node1.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

            if(!batchnormFwdWithVarianceFusionCheckTensorsLogErrors(
                   bnInfAttr, actAttr, opGraph.getTensorMap()))
            {
                return false;
            }

            try
            {
                validator.checkInferenceVarianceExtActivationTensorConfigSupported(bnInfAttr,
                                                                                   actAttr);
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

        const auto nodeAttrs = getBatchnormBackwardFusionNodeAttrsLogErrors(opGraph);
        if(!nodeAttrs.has_value())
        {
            return false;
        }

        if(!batchnormBwdFusionCheckTensorsLogErrors(std::get<0>(nodeAttrs.value()),
                                                    std::get<1>(nodeAttrs.value()),
                                                    std::get<2>(nodeAttrs.value()),
                                                    opGraph.getTensorMap()))
        {
            return false;
        }

        try
        {
            BatchnormValidator validator(opGraph.getTensorMap());
            validator.checkBwdActivationTensorConfigSupported(std::get<0>(nodeAttrs.value()),
                                                              std::get<1>(nodeAttrs.value()),
                                                              std::get<2>(nodeAttrs.value()));
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        HIPDNN_PLUGIN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                               "activation + batchnorm backward fusion");
        return true;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Batchnorm plan builder is applicable only for 1, 2 or 3 node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t BatchnormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanInferenceWithVarianceSingleNode(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& attr = nodeWrapper.attributesAs<
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();

    BatchnormFwdInferenceWithVarianceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferenceWithVariancePlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedFwdInferenceActivation(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);

    const auto& fwdInference
        = node0.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& activation
        = node1.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    BatchnormFwdInferenceParams params(fwdInference, activation, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedFwdInferenceWithVarianceActivation(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);

    const auto& fwdInference = node0.attributesAs<
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();
    const auto& activation
        = node1.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    BatchnormFwdInferenceWithVarianceParams params(
        fwdInference, activation, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferenceWithVariancePlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFwdTrainingSingleNode(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();

    BatchnormFwdTrainingParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBackwardsActivation(
    [[maybe_unused]] const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    HipKernelContext& executionContext)
{
    const auto [bnInfAttr, actAttr, bnBwdAttr] = getBatchnormBackwardFusionNodeAttrs(opGraph);
    batchnormBwdFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, opGraph.getTensorMap());

    BatchnormBwdParams params(bnBwdAttr, actAttr, bnInfAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void BatchnormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings) const
{
}

void BatchnormPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipKernelContext& executionContext) const
{
    if(opGraph.nodeCount() == 2)
    {
        const auto& node0 = opGraph.getNodeWrapper(0);
        if(node0.attributesType()
           == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes)
        {
            HIPDNN_PLUGIN_LOG_INFO("Building batchnorm inference + activation fusion plan");
            buildPlanFusedFwdInferenceActivation(
                handle, opGraph, _kernelCompiler, _devicePropertyProvider, executionContext);
        }
        else if(node0.attributesType()
                == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                    BatchnormInferenceAttributesVarianceExt)
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Building batchnorm inference with variance + activation fusion plan");
            buildPlanFusedFwdInferenceWithVarianceActivation(
                handle, opGraph, _kernelCompiler, _devicePropertyProvider, executionContext);
        }
        return;
    }

    if(opGraph.nodeCount() == 3)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Building batchnorm inference + activation + batchnorm backward fusion plan");
        buildPlanFusedBackwardsActivation(
            handle, opGraph, _kernelCompiler, _devicePropertyProvider, executionContext);
        return;
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm fwd training plan for node: " << nodeName);
        buildPlanFwdTrainingSingleNode(handle,
                                       opGraph,
                                       nodeWrapper,
                                       _kernelCompiler,
                                       _devicePropertyProvider,
                                       executionContext);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm fwd inference plan for node: " << nodeName);
        buildPlanInferenceSingleNode(handle,
                                     opGraph,
                                     nodeWrapper,
                                     _kernelCompiler,
                                     _devicePropertyProvider,
                                     executionContext);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        BatchnormInferenceAttributesVarianceExt:
        HIPDNN_PLUGIN_LOG_INFO(
            "Building batchnorm fwd inference with variance plan for node: " << nodeName);
        buildPlanInferenceWithVarianceSingleNode(handle,
                                                 opGraph,
                                                 nodeWrapper,
                                                 _kernelCompiler,
                                                 _devicePropertyProvider,
                                                 executionContext);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building batchnorm backward plan for node: " << nodeName);
        buildPlanBwdSingleNode(handle,
                               opGraph,
                               nodeWrapper,
                               _kernelCompiler,
                               _devicePropertyProvider,
                               executionContext);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> BatchnormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::batchnorm

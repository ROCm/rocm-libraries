// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <exception>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <set>
#include <string>

#include "RMSnormBwdPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlan.hpp"

namespace hip_kernel_provider::rmsnorm
{

namespace
{

void backwardActivationCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activationAttr,
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& bwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto activationIn1Uid = activationAttr.in_1_tensor_uid();
    if(!activationIn1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward requires in_1 tensor (forward activation input)");
    }

    if(activationAttr.out_0_tensor_uid() != bwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Rmsnorm backward dy input must be the activation output tensor");
    }

    const auto& activationTensorIn0
        = findTensorAttributes(tensorMap, activationAttr.in_0_tensor_uid());
    if(activationTensorIn0.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation in_0 (dy gradient) must be non-virtual");
    }

    const auto& actTensorIn1 = findTensorAttributes(tensorMap, activationIn1Uid.value());
    const auto& actTensorOut = findTensorAttributes(tensorMap, activationAttr.out_0_tensor_uid());
    if(actTensorIn1.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from rmsnorm must not be virtual, output must be virtual");
    }

    const auto& bwdTensorDy = findTensorAttributes(tensorMap, bwdAttr.dy_tensor_uid());
    const auto& bwdTensorDx = findTensorAttributes(tensorMap, bwdAttr.dx_tensor_uid());
    const auto& bwdTensorDscale = findTensorAttributes(tensorMap, bwdAttr.dscale_tensor_uid());
    const auto* bwdTensorDbias
        = bwdAttr.dbias_tensor_uid().has_value()
              ? &findTensorAttributes(tensorMap, bwdAttr.dbias_tensor_uid().value())
              : nullptr;

    if(!bwdTensorDy.virtual_() || bwdTensorDx.virtual_() || bwdTensorDscale.virtual_()
       || (bwdTensorDbias != nullptr && bwdTensorDbias->virtual_()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Rmsnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

} // namespace

RMSnormBwdPlanBuilder::RMSnormBwdPlanBuilder(const IKernelCompiler& kernelCompiler,
                                             const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool RMSnormBwdPlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the plan bakes those dims into the compiled kernel
    // launch, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO("RMSnormBwd plan builder does not support override shapes");
        return false;
    }

    auto anyNodeIsNotF32Compute = [&]() {
        return !std::all_of(
            opGraph.nodeWrappers().begin(), opGraph.nodeWrappers().end(), [](const auto& node) {
                return node->computeDataType()
                       == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
            });
    };

    switch(opGraph.nodeCount())
    {
    case 1: // Backward
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "RMSnorm backward plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                       RMSNormBackwardAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "RMSnorm backward plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkBwdTensorConfigSupported(
                *node.attributes_as_RMSNormBackwardAttributes());
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    case 2: // Backward fused activation (backward + activation)
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "RMSnorm backward plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        const auto& nodeWrapper0 = opGraph.getNodeWrapper(0);
        const auto& nodeWrapper1 = opGraph.getNodeWrapper(1);

        const bool isPointwiseFirst
            = nodeWrapper0.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;
        const bool isBwdSecond
            = nodeWrapper1.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes;

        if(!isPointwiseFirst || !isBwdSecond)
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "RMSnorm bwd plan builder is not applicable for this graph node order and types");
            return false;
        }

        try
        {
            const auto& node0 = opGraph.getNode(0);
            const auto& node1 = opGraph.getNode(1);
            backwardActivationCheckTensors(*node0.attributes_as_PointwiseAttributes(),
                                           *node1.attributes_as_RMSNormBackwardAttributes(),
                                           opGraph.getTensorMap());
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkBwdActivationTensorConfigSupported(
                *node0.attributes_as_PointwiseAttributes(),
                *node1.attributes_as_RMSNormBackwardAttributes());
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "RMSnorm backward plan builder is applicable only for 1 or 2 node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t RMSnormBwdPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    // RMSnorm backward plan currently does not require any workspace.
    return 0u;
}

void RMSnormBwdPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

namespace
{

void buildPlanBackward(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    Context& executionContext)
{
    const auto& attr
        = nodeWrapper
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes>();

    RMSnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanBackwardActivation(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapperActivation,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapperBwd,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    Context& executionContext)
{
    const auto& activationAttr
        = nodeWrapperActivation
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();
    const auto& bwdAttr
        = nodeWrapperBwd
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes>();
    backwardActivationCheckTensors(activationAttr, bwdAttr, opGraph.getTensorMap());

    RMSnormBwdParams params(bwdAttr, activationAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void RMSnormBwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    Context& executionContext) const
{
    switch(opGraph.nodeCount())
    {
    case 1:
    {
        const auto& nodeWrapper = opGraph.getNodeWrapper(0);
        const auto nodeName = nodeWrapper.name();

        HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm backward plan for node: " << nodeName);
        buildPlanBackward(handle,
                          opGraph,
                          nodeWrapper,
                          _kernelCompiler,
                          _devicePropertyProvider,
                          executionContext);

        break;
    }
    case 2:
    {
        const auto& nodeWrapper0 = opGraph.getNodeWrapper(0);
        const auto& nodeWrapper1 = opGraph.getNodeWrapper(1);
        const auto nodeName0 = nodeWrapper0.name();
        const auto nodeName1 = nodeWrapper1.name();

        HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm backward activation plan for nodes: "
                               << nodeName0 << ", " << nodeName1);
        buildPlanBackwardActivation(handle,
                                    opGraph,
                                    nodeWrapper0,
                                    nodeWrapper1,
                                    _kernelCompiler,
                                    _devicePropertyProvider,
                                    executionContext);

        break;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_ERROR("RMSnorm backward cannot build a plan for " << opGraph.nodeCount()
                                                                            << " nodes");
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Invalid number of nodes in RMSnorm backward plan builder");
    }
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> RMSnormBwdPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // hip_kernel_provider::rmsnorm

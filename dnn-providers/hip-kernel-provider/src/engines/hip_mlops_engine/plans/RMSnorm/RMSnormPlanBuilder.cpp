// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "RMSnormApplicabilityChecks.hpp"
#include "RMSnormFwdPlan.hpp"
#include "RMSnormPlanBuilder.hpp"

#include <algorithm>
#include <set>

namespace hip_kernel_provider::rmsnorm
{

namespace
{

void forwardActivationCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& fwdAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activationAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    if(activationAttr.in_0_tensor_uid() != fwdAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Rmsnorm forward y output must be the activation input tensor");
    }

    const auto& activationTensorIn0
        = findTensorAttributes(tensorMap, activationAttr.in_0_tensor_uid());
    if(!activationTensorIn0.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Activation in_0 (y) must be virtual");
    }

    const auto& activationTensorOut0
        = findTensorAttributes(tensorMap, activationAttr.out_0_tensor_uid());
    if(activationTensorOut0.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Activation output must be non-virtual");
    }

    const auto& fwdTensorX = findTensorAttributes(tensorMap, fwdAttr.x_tensor_uid());
    const auto& fwdTensorY = findTensorAttributes(tensorMap, fwdAttr.y_tensor_uid());
    const auto* fwdTensorInvRms
        = fwdAttr.inv_rms_tensor_uid().has_value()
              ? &findTensorAttributes(tensorMap, fwdAttr.inv_rms_tensor_uid().value())
              : nullptr;

    if(fwdTensorX.virtual_() || !fwdTensorY.virtual_()
       || (fwdTensorInvRms != nullptr && fwdTensorInvRms->virtual_()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Rmsnorm forward x input must not be virtual, y output tensor must be virtual, inverse "
            "RMS output tensor must not be virtual");
    }
}

} // namespace

RMSnormPlanBuilder::RMSnormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                       const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool RMSnormPlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the plan bakes those dims into the compiled kernel
    // launch, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO("RMSnorm plan builder does not support override shapes");
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
    case 1: // Forward
    {
        // Kernel code always uses fp32 compute type
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR("RMSnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("RMSnorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkFwdTensorConfigSupported(*node.attributes_as_RMSNormAttributes());
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    case 2: // Forward fused activation (forward + activation)
    {
        // Kernel code always uses fp32 compute type
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR("RMSnorm plan builder only supports nodes with an fp32 "
                                    "compute_data_type");
            return false;
        }

        const auto& nodeWrapper0 = opGraph.getNodeWrapper(0);
        const auto& nodeWrapper1 = opGraph.getNodeWrapper(1);

        const bool isFwdFirst
            = nodeWrapper0.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes;
        const bool isPointwiseSecond
            = nodeWrapper1.attributesType()
              == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;

        if(!isFwdFirst || !isPointwiseSecond)
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "RMSnorm fwd plan builder is not applicable for this graph node order and types");
            return false;
        }

        try
        {
            const auto& node0 = opGraph.getNode(0);
            const auto& node1 = opGraph.getNode(1);
            forwardActivationCheckTensors(*node0.attributes_as_RMSNormAttributes(),
                                          *node1.attributes_as_PointwiseAttributes(),
                                          opGraph.getTensorMap());
            rmsnorm::RMSnormValidator validator(opGraph.getTensorMap());
            validator.checkFwdActivationTensorConfigSupported(
                *node0.attributes_as_RMSNormAttributes(),
                *node1.attributes_as_PointwiseAttributes());
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
            "RMSnorm forward plan builder is applicable only for 1 or 2 node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t RMSnormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    // RMS norm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanForward([[maybe_unused]] const Handle& handle,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                      const IKernelCompiler& kernelCompiler,
                      const IDevicePropertyProvider& devicePropertyProvider,
                      Context& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes>();

    RMSnormFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormFwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

void buildPlanForwardActivation(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapperFwd,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapperActivation,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    Context& executionContext)
{
    const auto& fwdAttr
        = nodeWrapperFwd.attributesAs<hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes>();
    const auto& activationAttr
        = nodeWrapperActivation
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();
    forwardActivationCheckTensors(fwdAttr, activationAttr, opGraph.getTensorMap());

    RMSnormFwdParams params(fwdAttr, activationAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<RMSnormFwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void RMSnormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

void RMSnormPlanBuilder::buildPlan(
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

        HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm fwd plan for node: " << nodeName);
        buildPlanForward(handle,
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

        HIPDNN_PLUGIN_LOG_INFO("Building RMSnorm fwd plan for nodes: " << nodeName0 << ", "
                                                                       << nodeName1);
        buildPlanForwardActivation(handle,
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
        HIPDNN_PLUGIN_LOG_ERROR("RMSnorm forward cannot build a plan for " << opGraph.nodeCount()
                                                                           << " nodes");
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Invalid number of nodes in RMSnorm forward plan builder");
    }
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> RMSnormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "IDevicePropertyProvider.hpp"
#include "LayernormPlanBuilder.hpp"
#include "engines/hip_kernel_engine/plans/layernorm/LayernormApplicabilityChecks.hpp"
#include "engines/hip_kernel_engine/plans/layernorm/LayernormFwdPlan.hpp"
#include "hip/IKernelCompiler.hpp"

namespace hip_kernel_provider::layernorm
{

LayernormPlanBuilder::LayernormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                           const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool LayernormPlanBuilder::isApplicable(
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
            HIPDNN_PLUGIN_LOG_ERROR(
                "Layernorm plan builder only supports nodes with an fp32 compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("Layernorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            LayernormValidator validator(opGraph.getTensorMap());
            switch(node.attributes_type())
            {
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
                validator.checkTensorConfigSupported(*node.attributes_as_LayernormAttributes());
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
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Layernorm plan builder is only applicable to 1 node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t LayernormPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipKernelSettings& executionSettings) const
{
    // Layernorm plan builder does not require workspace size
    return 0;
}

namespace
{

void buildPlanFwd([[maybe_unused]] const HipKernelHandle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                  const IKernelCompiler& kernelCompiler,
                  const IDevicePropertyProvider& devicePropertyProvider,
                  HipKernelContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();

    LayernormFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<LayernormFwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());

    executionContext.setPlan(std::move(plan));
}

} // namespace

void LayernormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipKernelSettings& executionSettings) const
{
}

void LayernormPlanBuilder::buildPlan(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipKernelContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building layernorm fwd plan for node: " << nodeName);
    buildPlanFwd(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> LayernormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipKernelHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::layernorm

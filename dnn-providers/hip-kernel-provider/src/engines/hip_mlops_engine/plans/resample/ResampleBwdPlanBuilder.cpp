// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleBwdPlanBuilder.hpp"

#include "ResampleApplicabilityChecks.hpp"
#include "ResampleBwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <algorithm>
#include <set>

namespace hip_kernel_provider::resample
{

ResampleBwdPlanBuilder::ResampleBwdPlanBuilder(
    const IKernelCompiler& kernelCompiler, const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool ResampleBwdPlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
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
                "Resample backward plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Resample backward plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            resample::ResampleValidator validator(opGraph.getTensorMap());
            validator.checkBwdTensorConfigSupported(*node.attributes_as_ResampleBwdAttributes());
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
            "Resample backward plan builder is applicable only for single node graphs. "
            "Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t ResampleBwdPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    // Resample backward plan currently does not require any workspace.
    return 0u;
}

void ResampleBwdPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

namespace
{

void buildPlanSingleNode(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
    const IKernelCompiler& kernelCompiler,
    const IDevicePropertyProvider& devicePropertyProvider,
    Context& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes>();

    ResampleBwdParams params(attr, opGraph.getTensorMap(), nodeWrapper.computeDataType());
    auto plan = std::make_unique<ResampleBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

} // namespace

void ResampleBwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    Context& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building resample backward plan for node: " << nodeName);
    buildPlanSingleNode(
        handle, opGraph, nodeWrapper, _kernelCompiler, _devicePropertyProvider, executionContext);
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ResampleBwdPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::resample

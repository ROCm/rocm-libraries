// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResamplePlanBuilder.hpp"

#include "ResampleApplicabilityChecks.hpp"
#include "ResampleFwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>

namespace hip_kernel_provider::resample
{

namespace
{

// Saturates at INT64_MAX so that an overflowing shape is declined, not wrapped.
int64_t elementCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    int64_t count = 1;
    for(const auto dim : *tensor.dims())
    {
        if(dim <= 0)
        {
            return 0;
        }
        count = count > std::numeric_limits<int64_t>::max() / dim
                    ? std::numeric_limits<int64_t>::max()
                    : count * dim;
    }
    return count;
}

bool launchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& attr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto count = elementCount(*tensorMap.at(attr.y_tensor_uid()));
    for(const auto& shape : ResampleFwdPlan::launchShapes(count))
    {
        if(!launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock))
        {
            HIPDNN_PLUGIN_LOG_INFO("ResampleFwd kernel "
                                   << shape.kernelName << " needs " << shape.blocks << " blocks x "
                                   << shape.threadsPerBlock
                                   << " threads, which reaches the 2^32 thread limit");
            return false;
        }
    }
    return true;
}

} // namespace

ResamplePlanBuilder::ResamplePlanBuilder(const IKernelCompiler& kernelCompiler,
                                         const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool ResamplePlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "ResampleFwd plan builder is applicable only for single node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }

    const auto& nodeWrappers = opGraph.nodeWrappers();
    if(!std::all_of(nodeWrappers.begin(), nodeWrappers.end(), [](const auto& node) {
           return node->computeDataType() == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
       }))
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "ResampleFwd plan builder only supports nodes with an fp32 compute_data_type");
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
               hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes}))
    {
        HIPDNN_PLUGIN_LOG_INFO("ResampleFwd plan builder is not applicable for this graph");
        return false;
    }

    const auto& node = opGraph.getNode(0);
    try
    {
        ResampleValidator(opGraph.getTensorMap())
            .checkTensorConfigSupported(*node.attributes_as_ResampleFwdAttributes());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return launchShapesFitThreadLimit(*node.attributes_as_ResampleFwdAttributes(),
                                      opGraph.getTensorMap());
}

size_t ResamplePlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    return 0;
}

void ResamplePlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

void ResamplePlanBuilder::buildPlan(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    Context& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    HIPDNN_PLUGIN_LOG_INFO("Building ResampleFwd plan for node: " << nodeWrapper.name());

    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes>();
    ResampleFwdParams params(attr, opGraph.getTensorMap(), nodeWrapper.computeDataType());
    auto plan = std::make_unique<ResampleFwdPlan>(std::move(params));
    plan->compile(_kernelCompiler, _devicePropertyProvider.getDeviceProperties());
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ResamplePlanBuilder::getCustomKnobs(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::resample

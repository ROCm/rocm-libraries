// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ReferenceOpCoverage.hpp"

#include <algorithm>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_integration_tests::bundle
{

namespace
{

// Ops the CPU reference is required to handle.
//
// Keep this list honest: an entry here means bundles using that op are registered
// for CPU validation and will turn red if the reference cannot run them. Do not add
// an op speculatively.
const std::set<NodeAttributes>& cpuSupportedOps()
{
    static const std::set<NodeAttributes> s_ops = {
        NodeAttributes::BatchnormInferenceAttributes,
        NodeAttributes::BatchnormInferenceAttributesVarianceExt,
        NodeAttributes::BatchnormAttributes,
        NodeAttributes::BatchnormBackwardAttributes,
        NodeAttributes::LayernormAttributes,
        NodeAttributes::LayernormBackwardAttributes,
        NodeAttributes::RMSNormAttributes,
        NodeAttributes::RMSNormBackwardAttributes,
        NodeAttributes::PointwiseAttributes,
    };
    return s_ops;
}

// Ops the GPU reference is required to handle. Narrower than the CPU set: the GPU
// reference dispatches through a signature-keyed plan registry, so coverage is
// per-op-shape and grows only as plan builders are written.
const std::set<NodeAttributes>& gpuSupportedOps()
{
    static const std::set<NodeAttributes> s_ops = {
        NodeAttributes::ConvolutionFwdAttributes,
        NodeAttributes::SdpaAttributes,
    };
    return s_ops;
}

} // namespace

const std::set<NodeAttributes>& referenceSupportedOps(ReferenceExecutorType type)
{
    switch(type)
    {
    case ReferenceExecutorType::CPU:
        return cpuSupportedOps();
    case ReferenceExecutorType::GPU:
        return gpuSupportedOps();
    default:
        throw std::runtime_error("Unknown reference executor type");
    }
}

std::optional<std::set<NodeAttributes>> graphNodeTypes(const void* graphBuffer, size_t size)
{
    std::set<NodeAttributes> types;
    try
    {
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        for(uint32_t i = 0; i < graph.nodeCount(); ++i)
        {
            types.insert(graph.getNode(i).attributes_type());
        }
    }
    catch(const std::exception&)
    {
        // An unreadable graph is not "covered by every reference". Reporting an
        // empty set would make referenceCoversGraph() vacuously true and register
        // a test for a bundle nobody can run.
        return std::nullopt;
    }
    return types;
}

bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    const auto types = graphNodeTypes(graphBuffer, size);
    if(!types.has_value() || types->empty())
    {
        return false;
    }

    const auto& supported = referenceSupportedOps(type);
    return std::all_of(types->begin(), types->end(), [&supported](const auto nodeType) {
        return supported.count(nodeType) != 0;
    });
}

std::vector<std::string>
    uncoveredNodeTypes(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    const auto types = graphNodeTypes(graphBuffer, size);
    if(!types.has_value())
    {
        return {std::string(K_UNREADABLE_GRAPH)};
    }

    std::vector<std::string> uncovered;
    const auto& supported = referenceSupportedOps(type);
    for(const auto nodeType : *types)
    {
        if(supported.count(nodeType) == 0)
        {
            uncovered.emplace_back(
                hipdnn_flatbuffers_sdk::data_objects::EnumNameNodeAttributes(nodeType));
        }
    }
    return uncovered;
}

std::string formatUncoveredOps(const std::set<std::string>& uncoveredOps)
{
    if(uncoveredOps.empty())
    {
        return {};
    }

    std::string formatted = " (";
    const char* separator = "";
    for(const auto& op : uncoveredOps)
    {
        formatted += separator;
        formatted += op;
        separator = ", ";
    }
    formatted += ")";
    return formatted;
}

} // namespace hipdnn_integration_tests::bundle

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ReferenceOpCoverage.hpp"

#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>

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

bool isFp8(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return true;
    default:
        return false;
    }
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

std::optional<bool> graphUsesRaggedTensors(const void* graphBuffer, size_t size)
{
    try
    {
        // Keep getTensorMap() inside the try: GraphWrapper's constructor does not
        // throw on a bad buffer, throwIfNotValid() inside this accessor does.
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        return !hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(graph.getTensorMap());
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
}

std::optional<bool> graphUsesFp8Tensors(const void* graphBuffer, size_t size)
{
    try
    {
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        for(const auto& entry : graph.getTensorMap())
        {
            if(entry.second != nullptr && isFp8(entry.second->data_type()))
            {
                return true;
            }
        }
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
    return false;
}

std::vector<std::string>
    exclusionReasons(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    // Three walks of the same buffer, at registration time only.  Folding them into
    // one pass would mean hand-rolling the node and tensor traversals here instead of
    // reusing the queries the tests pin individually.
    const auto types = graphNodeTypes(graphBuffer, size);
    const auto ragged = graphUsesRaggedTensors(graphBuffer, size);
    const auto fp8 = graphUsesFp8Tensors(graphBuffer, size);
    if(!types.has_value() || !ragged.has_value() || !fp8.has_value())
    {
        return {std::string(K_UNREADABLE_GRAPH)};
    }

    std::vector<std::string> reasons;
    if(types->empty())
    {
        reasons.emplace_back(K_NO_NODES);
    }

    const auto& supported = referenceSupportedOps(type);
    for(const auto nodeType : *types)
    {
        if(supported.count(nodeType) == 0)
        {
            reasons.emplace_back(
                hipdnn_flatbuffers_sdk::data_objects::EnumNameNodeAttributes(nodeType));
        }
    }

    if(*ragged)
    {
        reasons.emplace_back(K_RAGGED_TENSORS);
    }
    // FP8 is GPU-only: no GPU plan builder registers an FP8 signature, and
    // GpuFpReferenceSdpa::fprop() takes no descale parameters at all, so FP8 is a
    // missing feature in the reference kernel rather than a missing registry line.
    // The CPU reference has BlockScaleDequantizePlan and may legitimately handle FP8
    // for ops in its own set, so excluding it there would over-reject.
    if(*fp8 && type == ReferenceExecutorType::GPU)
    {
        reasons.emplace_back(K_FP8_TENSORS);
    }
    return reasons;
}

bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    return exclusionReasons(type, graphBuffer, size).empty();
}

std::string formatExclusionReasons(const std::set<std::string>& reasons)
{
    if(reasons.empty())
    {
        return {};
    }

    std::string formatted = " (";
    const char* separator = "";
    for(const auto& reason : reasons)
    {
        formatted += separator;
        formatted += reason;
        separator = ", ";
    }
    formatted += ")";
    return formatted;
}

} // namespace hipdnn_integration_tests::bundle

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/TomlGuards.hpp"

// Tolerance resolution shared by both verification harnesses. Reads per-op,
// per-dtype numbers from TestTolerances.hpp and aggregates them into a single
// atol/rtol pair for each output tensor.
//
// TOML per-test overrides apply to engine output only. They are read from an
// engine's own config file, so they say how far that engine may drift — never how
// far our checked-in golden data may. Reference validation calls defaultTolerance()
// and can never be loosened by an engine's config. See ALMIOPEN-2216 for the
// DynamicTolerances upgrade path.

namespace hipdnn_integration_tests::tolerance
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

// Per-op tolerance for one node attribute type, at a fixed element type T.
// Maps a flatbuffer NodeAttributes tag onto the corresponding TestTolerances.hpp
// entry. Unknown ops fall back to a conservative 1e-3.
template <typename T>
inline float toleranceForNodeAttributes(data::NodeAttributes attrType)
{
    using NA = data::NodeAttributes;
    namespace tol = hipdnn_test_sdk::utilities;

    switch(attrType)
    {
    case NA::ConvolutionFwdAttributes:
        return tol::conv::getToleranceFwd<T>();
    case NA::ConvolutionBwdAttributes:
        return tol::conv::getToleranceBwd<T>();
    case NA::ConvolutionWrwAttributes:
        return tol::conv::getToleranceWrw<T>();
    case NA::BatchnormInferenceAttributes:
        return tol::batchnorm::getToleranceInference<T>();
    case NA::BatchnormInferenceAttributesVarianceExt:
        return tol::batchnorm::getToleranceInferenceWithVariance<T>();
    case NA::BatchnormAttributes:
        return tol::batchnorm::getToleranceTraining<T>();
    case NA::BatchnormBackwardAttributes:
        return tol::batchnorm::getToleranceBackward<T>();
    case NA::MatmulAttributes:
        return tol::matmul::getTolerance<T>();
    case NA::MoeGroupedMatmulAttributes:
        return tol::moe::getToleranceFwd<T>();
    case NA::MoeGroupedMatmulBwdAttributes:
        return tol::moe::getToleranceBwd<T>();
    case NA::ReductionAttributes:
        return tol::reduction::getTolerance<T>();
    case NA::RMSNormAttributes:
        return tol::rmsnorm::getTolerance<T>();
    case NA::PointwiseAttributes:
        return tol::pointwise::getTolerance<T>();
    case NA::LayernormAttributes:
    case NA::LayernormBackwardAttributes:
        return tol::layernorm::getTolerance<T>();
    case NA::SdpaAttributes:
    case NA::SdpaBackwardAttributes:
        return tol::sdpa::getToleranceFwd<T>();
    default:
        return 1e-3f;
    }
}

// Dispatch the element-type template on a runtime DataType.
inline float toleranceForNode(data::NodeAttributes attrType, data::DataType dataType)
{
    using DT = data::DataType;
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::half;

    switch(dataType)
    {
    case DT::FLOAT:
        return toleranceForNodeAttributes<float>(attrType);
    case DT::HALF:
        return toleranceForNodeAttributes<half>(attrType);
    case DT::BFLOAT16:
        return toleranceForNodeAttributes<bfloat16>(attrType);
    case DT::DOUBLE:
        return toleranceForNodeAttributes<float>(attrType);
    default:
        return 1e-3f;
    }
}

// Selects how per-node tolerances are reduced to one value for an output.
enum class TolerancePolicy
{
    // Loosest per-node tolerance in the graph. Conservative default: never
    // tighter than any single node, so it cannot manufacture a false failure.
    MAX_ACROSS_NODES,

    // Tolerance of the last non-Pointwise (output-producing) node. Tighter
    // than MAX on fused chains whose loosest op is not the output op.
    OUTPUT_OP_TOLERANCE,
};

// Max-across-nodes: the loosest per-node tolerance in the graph.
// Returns 1e-3 for a graph with no nodes.
inline float maxAcrossNodes(const fb::GraphWrapper& wrapper, data::DataType dataType)
{
    const auto nodeCount = wrapper.nodeCount();

    bool found = false;
    float maxTolerance = 0.0f;
    for(uint32_t i = 0; i < nodeCount; ++i)
    {
        const auto attrType = wrapper.getNode(i).attributes_type();
        const float nodeTolerance = toleranceForNode(attrType, dataType);
        maxTolerance = found ? std::max(maxTolerance, nodeTolerance) : nodeTolerance;
        found = true;
    }

    return found ? maxTolerance : 1e-3f;
}

// Output-op: tolerance of the last non-Pointwise node in topological order.
// Falls back to maxAcrossNodes if every node is Pointwise.
inline float outputOpTolerance(const fb::GraphWrapper& wrapper, data::DataType dataType)
{
    const auto nodeCount = wrapper.nodeCount();

    bool foundRoot = false;
    data::NodeAttributes rootAttr = data::NodeAttributes::NONE;
    for(uint32_t i = 0; i < nodeCount; ++i)
    {
        const auto attrType = wrapper.getNode(i).attributes_type();
        if(attrType != data::NodeAttributes::PointwiseAttributes)
        {
            rootAttr = attrType;
            foundRoot = true;
        }
    }

    if(!foundRoot)
    {
        return maxAcrossNodes(wrapper, dataType);
    }
    return toleranceForNode(rootAttr, dataType);
}

// The policy-selected default for one output tensor, with no per-test override.
//
// This is what a reference-vs-golden comparison uses. TOML overrides live in an
// engine's own config file (config/<ENGINE>.toml), so they describe how far *that
// engine* may drift; applying one to our own checked-in golden data would loosen
// the gate that is supposed to be measuring the data.
inline float defaultTolerance(const fb::GraphWrapper& wrapper,
                              data::DataType dataType,
                              TolerancePolicy policy = TolerancePolicy::MAX_ACROSS_NODES)
{
    switch(policy)
    {
    case TolerancePolicy::MAX_ACROSS_NODES:
        return maxAcrossNodes(wrapper, dataType);
    case TolerancePolicy::OUTPUT_OP_TOLERANCE:
        return outputOpTolerance(wrapper, dataType);
    default:
        throw std::invalid_argument("unknown TolerancePolicy");
    }
}

// Resolve atol/rtol for an output tensor an *engine* produced: policy-selected
// default, then the engine's TOML override.
inline void resolveTolerance(const fb::GraphWrapper& wrapper,
                             data::DataType dataType,
                             const std::string& testName,
                             float& atol,
                             float& rtol,
                             TolerancePolicy policy = TolerancePolicy::MAX_ACROSS_NODES)
{
    atol = defaultTolerance(wrapper, dataType, policy);
    rtol = atol;
    applyTomlToleranceOverride(testName, atol, rtol);
}

} // namespace hipdnn_integration_tests::tolerance

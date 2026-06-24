// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/TomlGuards.hpp"

// Shared default-tolerance resolution for both verification harnesses
// (ALMIOPEN-2216). Both the programmatic graph harness and the bundle harness
// reduce to the same question — "given a serialized graph and an output dtype,
// what default atol/rtol should the comparison use?" — so the policy lives here,
// keyed on the flatbuffer GraphWrapper, which is the common representation: the
// bundle harness already holds one, and the graph harness obtains it via
// Graph::to_binary().
//
// This header owns POLICY only; the per-operation / per-dtype tolerance NUMBERS
// stay in hipdnn_test_sdk TestTolerances.hpp and are read, never modified.
//
// TODO(dynamic tolerance): the per-op tolerance source here is the FIXED table
// (TestTolerances.hpp). The codebase already ships a dynamic, shape/dtype-aware
// model — hipdnn_test_sdk DynamicTolerances.hpp + per-op DynamicTolerances{Matmul,
// Conv,BatchNorm,LayerNorm,RMSNorm,Sdpa,Pointwise}.hpp and
// pointwise/PointwiseErrorClassification.hpp — already wired into other test
// fixtures (conv gpu-ref, sdpa backward, cpu-executor plan tests). RFC 0011
// §"Tolerance Framework" / "Future Work #1" defines the upgrade: replace the
// fixed level-3 default with DynamicTolerances, keyed on graph properties
// (op, dtype, tensor dims), without changing the three-level chain or this
// aggregation policy. When promoting, add a sibling aggregation function that
// routes through the existing DynamicTolerances functions instead of
// TestTolerances.hpp, and pass it to resolveTolerance; also add
// sub-bf16 dtypes (FP4) which the current DataType switch lacks (falls through to
// 1e-3). See ALMIOPEN-2216.
//
// Two policy decisions are encoded here, each kept independently evolvable:
//
//   * Aggregation = max-across-nodes. The output tolerance is the loosest
//     per-node tolerance in the graph. This is the conservative envelope: it can
//     be too loose on a long fused chain but is never too tight, so it never
//     manufactures a false failure. Root-op-only selection (the graph harness's
//     prior heuristic) is unsafe — an upstream high-K / low-precision node
//     dominates the error, so picking the "root" can under-tolerance and fail a
//     correct kernel. A principled alternative (analytic error propagation along
//     the producer chain) is the documented future upgrade; it needs per-op
//     condition-number models and is deferred.
//
//   * dtype key = the OUTPUT tensor's dtype (passed in by the caller). Truly
//     per-node dtype keying — each node keyed on its own output-edge dtype — only
//     differs from this in mixed-I/O fused graphs, and recovering a node's
//     output dtype needs a per-op tensor-UID extractor (the flatbuffer Node
//     carries only compute_data_type, not its I/O tensors). That extractor is
//     the same machinery the per-output subgraph walk needs, so per-node dtype is
//     deferred together with multi-output support (ALMIOPEN-2216).
//
// resolveTolerance() is the single entry point for both harnesses: it derives
// the max-across-nodes default and then applies the TOML per-test override (the
// highest-priority layer) in one place, so neither harness applies the override
// separately and the layering order lives here alone.

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
    case NA::ReductionAttributes:
        return tol::reduction::getTolerance<T>();
    case NA::RMSNormAttributes:
        return tol::rmsnorm::getTolerance<T>();
    case NA::PointwiseAttributes:
        return tol::pointwise::getTolerance<T>();
    case NA::LayernormAttributes:
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
    default:
        return 1e-3f;
    }
}

// An aggregation policy reduces the per-node tolerances of a graph to one
// default tolerance for an output. It is just a function (graph, dtype) -> float;
// new policies are added as new functions, and resolveTolerance() takes the
// chosen one as a parameter. No enum/switch — the policy IS the function.
using AggregationPolicy = float (*)(const fb::GraphWrapper&, data::DataType);

// Conservative policy (the default): max-across-nodes — the loosest per-node
// tolerance in the graph. Never tighter than any single node, so it cannot
// manufacture a false failure; for a fused output (which genuinely accumulates
// error from every op on its chain) the loosest contributing op is the correct
// floor. Returns 1e-3 for a graph with no nodes.
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

// Output-op policy: the tolerance of the last non-Pointwise node in topological
// order — i.e. the op that produces the graph's output. This reproduces the
// graph harness's historical getTolerance() behavior so the C++ graph tests keep
// their exact tolerances as they migrate. It is tighter than maxAcrossNodes only
// on fused chains whose loosest op is NOT the output op; for the common case
// (one real op + activation) the two policies are identical, since the activation
// is Pointwise (skipped) and the single real op is both loosest and last.
//
// NOTE: this is a heuristic, not a principled tight bound — it attributes the
// whole output's tolerance to one op and ignores upstream error accumulation.
// Kept only for migration parity; max remains the default everywhere else, and
// the principled tighten path is the future DynamicTolerances upgrade. Falls back
// to maxAcrossNodes if every node is Pointwise (no clear producing op).
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
            rootAttr = attrType; // last non-Pointwise wins (topological order)
            foundRoot = true;
        }
    }

    if(!foundRoot)
    {
        return maxAcrossNodes(wrapper, dataType);
    }
    return toleranceForNode(rootAttr, dataType);
}

// Future policies live here as sibling functions, e.g.:
//   float propagatedBound(wrapper, dtype);    // analytic error propagation
//   float dynamic(wrapper, dtype);            // wired to DynamicTolerances.hpp
// Each is added without touching resolveTolerance or any caller — pass it in.

// Warn (once per call site) when a graph has more than one output tensor.
//
// Every current aggregation policy reduces over the WHOLE graph, not the subgraph
// that produces a given output: maxAcrossNodes takes the loosest of all nodes,
// outputOpTolerance takes the single last non-Pointwise node. For a multi-output
// graph neither is scoped to the output being toleranced, so a tolerance may be
// attributed from an unrelated branch. The precise fix (per-output subgraph
// scoping) is deferred together with per-node dtype keying (ALMIOPEN-2216),
// because both need a per-op tensor-UID extractor. Until then we surface the
// imprecision loudly rather than letting it pass silently.
inline void warnIfMultipleOutputs(std::size_t outputCount, const char* context)
{
    if(outputCount > 1)
    {
        HIPDNN_PLUGIN_LOG_WARN(context
                               << ": graph has " << outputCount
                               << " output tensors; tolerance is reduced over the whole graph, not "
                                  "the per-output subgraph (deferred, ALMIOPEN-2216)");
    }
}

// Resolve the FINAL absolute/relative tolerance for an output tensor of the
// given dtype: the chosen aggregation policy's default (max-across-nodes unless
// overridden), then the TOML per-test override (highest priority) applied on top.
// This is the single tolerance entry point for both harnesses — neither applies
// the override separately, so the layering order (default -> override) lives in
// exactly one place. The aggregation policy is a parameter (default
// maxAcrossNodes) so a caller can select a different policy without any change
// here.
inline void resolveTolerance(const fb::GraphWrapper& wrapper,
                             data::DataType dataType,
                             const std::string& testName,
                             float& atol,
                             float& rtol,
                             AggregationPolicy aggregate = maxAcrossNodes)
{
    const float defaultTolerance = aggregate(wrapper, dataType);
    atol = defaultTolerance;
    rtol = defaultTolerance;
    applyTomlToleranceOverride(testName, atol, rtol);
}

} // namespace hipdnn_integration_tests::tolerance

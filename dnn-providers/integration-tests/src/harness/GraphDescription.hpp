// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sstream>
#include <string>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>

#include "NodeTypeNames.hpp"

namespace hipdnn_integration_tests
{

// Structured view of a graph's identifying features, used by the support
// claim verifier to match observations against [[supported.matchers]]
// entries. Producing structured fields rather than re-parsing the
// human-readable description avoids the test-name glob hazards described
// in RFC 0012 §11.4.
struct StructuredGraphDescription
{
    // Op chain only — e.g., "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD".
    // No dtype suffix.
    std::string opChain;
    // IO dtype string from to_string(DataType) — e.g., "fp16", "fp32", "bf16".
    std::string ioDtype;
    // Compute dtype — same value space as ioDtype.
    std::string computeDtype;
    // Intermediate dtype, or empty if graph_attributes has NOT_SET.
    std::string intermediateDtype;
};

// Per-node "variant" tag — returned by describeNodeVariant when a node's
// attribute set affects MIOpen solver dispatch in a way the bare node
// type can't capture. Producing distinct op_chain strings for these
// variants prevents the RFC 0012 §7 condenser's S∩U collision: without
// the tag, two structurally different graphs (e.g. training Batchnorm
// with vs. without running-stats inputs) serialize identically and the
// engine's per-variant support classification gets lost.
//
// Empty string means no variant — the node has no dispatch-affecting
// attribute variation we care about (yet).
inline std::string describeNodeVariant(const hipdnn_frontend::graph::INode& node)
{
    using namespace hipdnn_frontend::graph;

    if(const auto* pw = dynamic_cast<const PointwiseNode*>(&node))
    {
        // PointwiseMode (RELU_FWD, SIGMOID, ...) is the base variant.
        // Optional clip/slope parameters change the *math* of the
        // operation even within the same mode: RELU_FWD with no params
        // is plain ReLU; with upper_clip=6.0 it's ReLU6; with both
        // clips it's a CLAMP; with a lower_clip_slope it's leaky ReLU.
        // MIOpen dispatches these to different solvers, so they need
        // distinct op_chain strings or they collide in S∩U during the
        // condenser run (RFC 0012 §7).
        //
        // We encode which optional params are *set* (not their values
        // — different values within the same set typically share a
        // solver). Order is fixed and alphabetical to keep the variant
        // tag deterministic across runs.
        std::string variant(to_string(pw->attributes.get_mode()));
        std::string flags;
        const auto appendFlag = [&](const char* name) {
            if(!flags.empty())
            {
                flags += ",";
            }
            flags += name;
        };
        if(pw->attributes.get_elu_alpha().has_value())
        {
            appendFlag("elu_alpha");
        }
        if(pw->attributes.get_relu_lower_clip().has_value())
        {
            appendFlag("lower_clip");
        }
        if(pw->attributes.get_relu_lower_clip_slope().has_value())
        {
            appendFlag("lower_slope");
        }
        if(pw->attributes.get_relu_upper_clip().has_value())
        {
            appendFlag("upper_clip");
        }
        if(pw->attributes.get_softplus_beta().has_value())
        {
            appendFlag("softplus_beta");
        }
        if(pw->attributes.get_swish_beta().has_value())
        {
            appendFlag("swish_beta");
        }
        if(!flags.empty())
        {
            variant += "[" + flags + "]";
        }
        return variant;
    }
    if(const auto* red = dynamic_cast<const ReductionNode*>(&node))
    {
        auto mode = red->attributes.get_mode();
        return mode.has_value() ? std::string(to_string(*mode)) : std::string{};
    }
    // BatchnormNode intentionally has no variant tag. An earlier attempt
    // distinguished FULL_TRAINING vs WITH_BATCH_STATS via the presence of
    // prev_running_stats inputs, on the hypothesis that MIOpen would
    // dispatch the two topologies to different solvers. The regenerated
    // sidecar showed both variants landing in the same matcher with the
    // same coverage rectangle — MIOpen evidently treats them identically
    // on RDNA3 (the optional inputs are a no-op when not consumed). The
    // variant was non-load-bearing and added matcher-set noise. Rule of
    // thumb: a variant tag belongs here only when a real S∩U conflict
    // has demonstrated the bare node type is too coarse.
    return {};
}

// Build the structured description from a graph. Visit order and node
// stringification match describeGraph() byte-for-byte; the two functions
// are co-designed so the composed description equals describeGraph()'s
// output.
inline StructuredGraphDescription
    describeGraphStructured(const hipdnn_frontend::graph::Graph& graph)
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    StructuredGraphDescription result;

    std::ostringstream ops;
    bool first = true;

    graph.visit([&](const INode& node) {
        // Skip the Graph root node itself
        if(dynamic_cast<const Graph*>(&node) != nullptr)
        {
            return;
        }

        if(!first)
        {
            ops << " + ";
        }
        first = false;

        ops << to_string(node.getNodeType());

        const auto variant = describeNodeVariant(node);
        if(!variant.empty())
        {
            ops << ":" << variant;
        }
    });

    result.opChain = ops.str();
    result.ioDtype = to_string(graph.graph_attributes.get_io_data_type());
    result.computeDtype = to_string(graph.graph_attributes.get_compute_data_type());

    if(graph.graph_attributes.get_intermediate_data_type() != DataType::NOT_SET)
    {
        result.intermediateDtype = to_string(graph.graph_attributes.get_intermediate_data_type());
    }

    return result;
}

// Compose the structured description back into the historical single-string
// form, e.g. "ConvFprop + Pointwise:RELU_FWD [io=fp16, compute=fp32, intermediate=fp16]".
// The format is stable — see RFC 0012 §12 "describeGraph output format" risk.
inline std::string describeGraph(const StructuredGraphDescription& desc)
{
    std::ostringstream out;
    out << desc.opChain << " [io=" << desc.ioDtype << ", compute=" << desc.computeDtype;
    if(!desc.intermediateDtype.empty())
    {
        out << ", intermediate=" << desc.intermediateDtype;
    }
    out << "]";
    return out.str();
}

inline std::string describeGraph(const hipdnn_frontend::graph::Graph& graph)
{
    return describeGraph(describeGraphStructured(graph));
}

} // namespace hipdnn_integration_tests

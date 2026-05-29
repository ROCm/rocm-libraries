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

        // For Pointwise nodes, append the mode
        if(const auto* pw = dynamic_cast<const PointwiseNode*>(&node))
        {
            ops << ":" << to_string(pw->attributes.get_mode());
        }
        // For Reduction nodes, append the mode
        else if(const auto* red = dynamic_cast<const ReductionNode*>(&node))
        {
            auto mode = red->attributes.get_mode();
            if(mode.has_value())
            {
                ops << ":" << to_string(*mode);
            }
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

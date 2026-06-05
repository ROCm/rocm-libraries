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

// Produces a compact description of a graph's operations and data types.
// Example: "ConvFprop + Pointwise:RELU_FWD [io=fp16, compute=fp32, intermediate=fp16]"
inline std::string describeGraph(const hipdnn_frontend::graph::Graph& graph)
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

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

    // Append data type context
    ops << " [io=" << to_string(graph.graph_attributes.get_io_data_type())
        << ", compute=" << to_string(graph.graph_attributes.get_compute_data_type());

    if(graph.graph_attributes.get_intermediate_data_type() != DataType::NOT_SET)
    {
        ops << ", intermediate=" << to_string(graph.graph_attributes.get_intermediate_data_type());
    }

    ops << "]";

    return ops.str();
}

} // namespace hipdnn_integration_tests

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend.hpp>

namespace hipdnn_tests
{

/// Exposes protected Graph methods needed by lifting integration tests.
/// Used by tests that lower a graph and then lift it back with fromBackendDescriptor().
class TestableGraphLifting : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::fromBackendDescriptor;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<hipdnn_frontend::graph::INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

/// Exposes protected Graph methods needed by lowering integration tests.
/// Used by tests that lower a graph via build_operation_graph_via_descriptors()
/// and then retrieve the serialized graph for verification.
class TestableGraphLowering : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

/// Exposes protected Graph methods needed by knob lifting tests.
/// Used by tests that exercise the descriptor-based knob lifting path.
class TestableGraphKnobs : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::get_knobs_for_engine_via_descriptors;
};

/// Exposes protected Graph methods needed by knob lowering tests.
/// Used by tests that lower knob settings via descriptors and
/// create execution plans from them.
class TestableGraphKnobLowering : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
};

} // namespace hipdnn_tests

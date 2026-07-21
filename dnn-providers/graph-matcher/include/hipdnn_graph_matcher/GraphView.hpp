// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// GraphView: precomputed producer/consumer adjacency over an IGraph, built once
// per match session via the OpSchema registry. The FlatBuffer graph exposes no
// producer/consumer index (edges are implicit tensor UIDs), so the matcher
// cannot walk up (merge/join) or down (fan-out) without this. Consumer lists
// keep every operand occurrence, so use-count (operand slots referencing a UID)
// and consumer-count (distinct consumer nodes) are both answerable -- a
// distinction that is load-bearing for fusion legality.

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/OpSchema.hpp>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hipdnn::graph_matcher {

using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;

// One end of an edge: which node, which role of that node, and which slot within
// the role (>0 only for variadic roles). `roleIndex` indexes the schema's
// operands[] (for a consumer) or results[] (for a producer).
struct Endpoint {
    uint32_t nodeIndex;
    uint32_t roleIndex;
    uint32_t slot;
};

class GraphView {
   public:
    // Builds adjacency over `graph` using `registry` (defaults to the builtin
    // one). The view is read-only and borrows the graph; the graph must outlive
    // it. Nodes with no registered schema are skipped (their edges are unknown).
    explicit GraphView(const IGraph& graph,
                       const OpSchemaRegistry& registry = OpSchemaRegistry::builtin());

    const IGraph& graph() const noexcept {
        return _graph;
    }

    // The single producing endpoint of a tensor UID, or nullptr if the UID has
    // no producer in this graph (i.e. it is a graph input / leaf).
    const Endpoint* producerOf(int64_t uid) const noexcept;

    // All consuming endpoints of a tensor UID, one per operand slot that
    // references it (so a UID used twice by one node appears twice). Empty span
    // if unconsumed. The returned pointer is valid for the view's lifetime.
    const std::vector<Endpoint>& consumersOf(int64_t uid) const noexcept;

    // Number of operand slots referencing a UID (XLA "use-count"): the
    // fusion-legality quantity. `mul($h, $h)` yields 2.
    size_t useCount(int64_t uid) const noexcept {
        return consumersOf(uid).size();
    }

    // Number of *distinct* consumer nodes of a UID (XLA "user-count").
    // `mul($h, $h)` yields 1.
    size_t consumerNodeCount(int64_t uid) const noexcept;

    // Tensor attributes for a UID, or nullptr if absent from the graph.
    const TensorAttributes* tensor(int64_t uid) const noexcept;

    // The operand UIDs of a node in role/slot order (schema operands flattened).
    std::vector<int64_t> operandUids(uint32_t nodeIndex) const;

    // The result UIDs of a node in role/slot order (schema results flattened).
    std::vector<int64_t> resultUids(uint32_t nodeIndex) const;

    // Canonical opcode of a node, or "" if the node has no registered schema.
    std::string_view opcodeOf(uint32_t nodeIndex) const;

    // Schema for a node (nullptr if unregistered). Lets a caller resolve role
    // names to indices without re-touching the registry.
    const OpSchema* schemaOf(uint32_t nodeIndex) const noexcept;

    // UID(s) contributed by one role of a node. `result` selects results[] over
    // operands[]; `roleIndex` indexes that role list. Empty if out of range or
    // the node is unregistered. Optional roles yield 0 or 1; variadic 0..N.
    std::vector<int64_t> roleUids(uint32_t nodeIndex, bool result, uint32_t roleIndex) const;

    // Scalar attribute of a node read as int64 (enum/bool/int), or nullopt if
    // the op has no such attribute or the field is absent.
    std::optional<int64_t> attrInt(uint32_t nodeIndex, std::string_view attrName) const;

   private:
    const IGraph& _graph;
    const OpSchemaRegistry& _registry;

    std::unordered_map<int64_t, Endpoint> _producers;
    std::unordered_map<int64_t, std::vector<Endpoint>> _consumers;

    static const std::vector<Endpoint> kEmptyEndpoints;
};

}  // namespace hipdnn::graph_matcher

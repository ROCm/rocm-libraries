// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// GPU-free unit-test graph helper for the autotune match-key selector
// (hipdnn_frontend::detail::getAutotuneConfigMatchKey).
//
// This is a DIFFERENT layer than GraphExecuteTestKit: there is no device, no
// buffers, no variant pack. A selector unit test only needs (1) a real op graph
// and (2) UIDs on its physical tensors, because the selector filters on
// has_uid() && !get_is_virtual() and a pre-build graph has no UIDs assigned.
//
// The helper does NOT hand-roll graph structure: it delegates to the existing
// FrontendGraphFactory (which already builds 15 op graphs), then walks the
// graph's node tree via the public INode::visit() + getNode*TensorAttributes()
// API (same pattern GraphExecuteTestKit uses) to ASSIGN UIDs to each physical
// (non-virtual) tensor and REPORT them back keyed by the factory-set tensor
// NAME ("x"/"w"/"dy"/...). A selector unit test then DISCOVERS each tensor by
// role (its name), reads the assigned UID + dims/strides, and forms its EXPECT
// values from those discovered UIDs rather than dictating them. UID assignment
// is op-agnostic, so a future deferred op's unit test reuses this verbatim.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>

namespace hipdnn_test_sdk::utilities
{

namespace graph = hipdnn_frontend::graph;

/// A graph created for a selector unit test, with UIDs already assigned to its
/// physical tensors and those tensors discoverable BY ROLE (the factory-set
/// name). Owns the Graph so the reported shared_ptr<TensorAttributes> stay
/// alive for the test's lifetime.
class SelectorUnitGraph
{
public:
    /// Build the op graph (delegating structure to FrontendGraphFactory) and
    /// assign a distinct UID to every physical (non-virtual) tensor.
    ///
    /// UIDs are assigned in graph enumeration order starting at @p uidBase and
    /// increasing by @p uidStep. The scheme is arbitrary: a selector unit test
    /// DISCOVERS each tensor's UID by role (byName) and forms its EXPECT values
    /// from the discovered value, so the specific numbers do not matter.
    explicit SelectorUnitGraph(OperationType op, int64_t uidBase = 1000, int64_t uidStep = 100)
        : _graph(FrontendGraphFactory::create(op))
    {
        int64_t nextUid = uidBase;
        _graph.visit([&](graph::INode& node) {
            const auto assign = [&](const std::shared_ptr<graph::TensorAttributes>& tensorAttr) {
                if(!tensorAttr || tensorAttr->get_is_virtual())
                {
                    return;
                }
                // A tensor shared across nodes is visited once: skip if
                // already assigned (also keeps the name map one-to-one).
                if(tensorAttr->has_uid())
                {
                    return;
                }
                tensorAttr->set_uid(nextUid);
                nextUid += uidStep;
                _byName.emplace(tensorAttr->get_name(), tensorAttr);
            };

            for(const auto& t : node.getNodeInputTensorAttributes())
            {
                assign(t);
            }
            for(const auto& t : node.getNodeOutputTensorAttributes())
            {
                assign(t);
            }
        });
    }

    /// The underlying graph (its root INode), to pass to the selectors.
    graph::Graph& graph()
    {
        return _graph;
    }
    const graph::Graph& graph() const
    {
        return _graph;
    }

    /// Look up a physical tensor by its role (factory-set name, e.g. "x"/"w"/
    /// "dy"). Throws if no such named tensor exists so a test mistake surfaces
    /// loudly rather than silently passing on a default.
    const std::shared_ptr<graph::TensorAttributes>& byName(const std::string& name) const
    {
        const auto it = _byName.find(name);
        if(it == _byName.end())
        {
            throw std::runtime_error("SelectorUnitGraph: no physical tensor named '" + name + "'");
        }
        return it->second;
    }

    /// Discovered UID for the named tensor.
    int64_t uidOf(const std::string& name) const
    {
        return byName(name)->get_uid();
    }

    /// All discovered physical tensors keyed by role/name.
    const std::unordered_map<std::string, std::shared_ptr<graph::TensorAttributes>>& byName() const
    {
        return _byName;
    }

private:
    graph::Graph _graph;
    std::unordered_map<std::string, std::shared_ptr<graph::TensorAttributes>> _byName;
};

} // namespace hipdnn_test_sdk::utilities

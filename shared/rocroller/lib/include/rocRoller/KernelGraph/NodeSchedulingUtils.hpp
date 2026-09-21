// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller::KernelGraph
{
    namespace NodeScheduling
    {
        /**
         * Returns all control nodes in graph.control that satisfy `pred`, grouped by immediate
         * body-parent.
         */
        std::unordered_map<int, std::vector<int>> getGroupedNodes(KernelGraph const&       graph,
                                                                  std::predicate<int> auto pred);

        /**
         * Returns all control nodes in graph.control of type `T`, grouped by immediate
         * body-parent.
         */
        template <ControlGraph::COperation T>
        std::unordered_map<int, std::vector<int>> getGroupedNodes(KernelGraph const& graph);

        /**
         * Creates a sub-graph of the given nodes.
         *
         * The sub-graph is created by adding the given nodes to a new control graph, and then
         * adding edges between the nodes based on the order of the nodes in the original
         * control graph.
         * 
         * rv.compare(cacheMode, a, b) should always return the same result as the original
         * control graph as long as a and b are both in `nodes`.
         */
        ControlGraph::ControlGraph createSubGraph(KernelGraph const&      graph,
                                                  std::vector<int> const& nodes);

        /**
         * Sorts `nodes` according to existing order and according to `comp`.
         *
         * `nodes` must be a collection of nodes directly within the same body-parent in
         * `graph.control`.
         */
        void orderNodes(KernelGraph const& graph, std::vector<int>& nodes, auto const& comp);
    }
}

#include "NodeSchedulingUtils_impl.hpp"

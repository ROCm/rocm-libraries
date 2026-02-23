
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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

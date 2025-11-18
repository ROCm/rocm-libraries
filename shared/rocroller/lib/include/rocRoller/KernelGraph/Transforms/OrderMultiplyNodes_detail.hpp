/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include "OrderMultiplyNodes.hpp"

#include <optional>
#include <unordered_map>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace OrderMultiplyNodesDetail
        {
            /**
             * Returns multiply nodes in graph grouped by their body parent.
             */
            std::unordered_map<int, std::vector<int>>
                getGroupedMultiplyNodes(KernelGraph const& graph);

            /**
             * Comparator for ordering multiply nodes.
             *
             * This is used to order the multiply nodes in each group.
             *
             * The order is determined by the following criteria:
             * 1. If the nodes are already ordered, use that order.
             * 2. Otherwise if available, use downstream memory nodes, to enable memory nodes
             *    to be scheduled earlier in some kernels.
             * 3. Otherwise if available, use last upstream tag dependencies, to prioritize
             *    multiplies that will have lower waitcount values.
             * 4. Otherwise use integer comparison as a last resort.
             */
            struct BestNodeOrder
            {
                BestNodeOrder(KernelGraph const& graph);

                bool operator()(int a, int b) const;

                std::optional<bool> existingOrder(int a, int b) const;
                std::optional<bool> orderByDownstreamMemoryNodes(int a, int b) const;
                std::optional<bool> orderByLastTagDependencies(int a, int b) const;

            private:
                std::optional<int>      downstreamMemoryNode(int node) const;
                std::vector<int> const& reversedTagDependencies(int node) const;

                KernelGraph const&                                  m_graph;
                ControlFlowRWTracer                                 m_tracer;
                mutable std::unordered_map<int, std::optional<int>> m_downstreamMemoryNodes;
                mutable std::unordered_map<int, std::vector<int>>   m_reversedTagDependencies;
            };

            struct OrderByDownstreamMemoryNodes
            {
                OrderByDownstreamMemoryNodes(KernelGraph const& graph);

                bool operator()(int a, int b) const;

            private:
                std::optional<int> downstreamMemoryNode(int node) const;

                KernelGraph const&                                  m_graph;
                mutable std::unordered_map<int, std::optional<int>> m_downstreamMemoryNodes;
            };

            struct OrderByLastTagDependencies
            {
                OrderByLastTagDependencies(KernelGraph const& graph);

                bool operator()(int a, int b) const;

            private:
                std::vector<int> const& reversedTagDependencies(int node) const;

                KernelGraph const&                                m_graph;
                ControlFlowRWTracer                               m_tracer;
                mutable std::unordered_map<int, std::vector<int>> m_reversedTagDependencies;
            };
        }
    }
}

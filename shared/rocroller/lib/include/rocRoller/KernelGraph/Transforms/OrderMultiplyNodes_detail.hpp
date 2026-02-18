// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "OrderMultiplyNodes.hpp"

#include <optional>
#include <unordered_map>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller::KernelGraph
{
    namespace OrderMultiplyNodesDetail
    {
        /**
         * Comparator for ordering multiply nodes.
         *
         * This is used to order the multiply nodes in each group.
         *
         * Note that this is NOT guaranteed to be a strict weak ordering and should not be
         * used with `std::sort`.
         *
         * The order is determined by the following criteria:
         * 1. If available, use downstream memory nodes, to enable memory nodes
         *    to be scheduled earlier in some kernels.
         * 2. Otherwise if available, use last upstream tag dependencies, to prioritize
         *    multiplies that will have lower waitcount values.
         * 3. Otherwise use integer comparison as a last resort.
         */
        struct BestNodeOrder
        {
            BestNodeOrder(KernelGraph const& graph);

            bool operator()(int a, int b) const;

            /**
             * Looks for the LoadLDSTile nodes that write the LHS and LHS_SCALE inputs ("A")
             * to `node` within the same loop and returns them in topological order.
             */
            std::vector<int> const& aTagReplacements(int node) const;

            /**
             * Orders by which LoadLDSTile nodes write the "A" inputs to `a` and `b` within the
             * same loop.  This will mean that we get a whole row of Multiply nodes first (i.e.
             * the Multiply nodes that use the same A input will be put first).
             */
            std::optional<bool> orderByATagReplacements(int a, int b) const;

            /**
             * Looks for memory nodes downstream of `node` in a breadth-first search of the
             * control graph. Returns the first memory node found, or `std::nullopt` if none
             * exists.
             */
            std::optional<int> downstreamMemoryNode(int node) const;

            /**
             * Looks for memory nodes downstream in a breadth-first search of the control
             * graph from `a` and `b`. If these are found and are different from each other,
             * it will return an order based on the control graph order of those memory nodes.
             *
             * For kernels that are not double buffered, this can be used to prioritize
             * multiply nodes that will enabled memory nodes to be scheduled earlier in the
             * kernel, overlapping with other multiply nodes.
             *
             * For double-buffered kernels, this will generally be the same memory node for
             * both `a` and `b`, so this will return `std::nullopt`.
             */
            std::optional<bool> orderByDownstreamMemoryNodes(int a, int b) const;

            /**
             * Looks for data flow tags that are read by `node`. Then looks for control
             * nodes that write to those tags, and are before `node`. Returns those nodes in
             * reverse topological order (i.e. from latest to earliest).
             */
            std::vector<int> const& reversedTagDependencies(int node) const;

            /**
             * Calls `reversedTagDependencies()` on `a` and `b`. Then will do an
             * element-wise comparison using `compareNodes()` on those results.
             */
            std::optional<bool> orderByLastTagDependencies(int a, int b) const;

            void logTagData() const;

            void populateCache(auto range) const;

        private:
            /**
             * Helper function. Returns the existing order of the two nodes if it is
             * defined, otherwise returns std::nullopt.
             */
            std::optional<bool> existingOrder(int a, int b) const;

            KernelGraph const&                                  m_graph;
            ControlFlowRWTracer                                 m_tracer;
            mutable std::unordered_map<int, std::optional<int>> m_downstreamMemoryNodes;
            mutable std::unordered_map<int, std::vector<int>>   m_reversedTagDependencies;
            mutable std::map<int, std::vector<int>>             m_aTagReplacements;
        };
    }
}

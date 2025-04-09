/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <functional>
#include <variant>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_fwd.hpp>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Graph/Hypergraph_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlEdge.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller
{
    /**
     * Control flow routines.
     *
     * Control flow is represented as a graph.  Nodes in the control flow graph represent
     * operations (like load/store or a for loop).  Edges in the graph encode dependencies
     * between nodes.
     *
     * The graph answers the question:
     * What are the series of operations needed to solve the problem?
     *
     * Each node of the graph, when traversed, will help generate assembly code.
     * It relies on coordinate transform graph expressions.
     *
     */
    namespace KernelGraph::ControlGraph
    {
        enum class NodeOrdering
        {
            LeftFirst = 0,
            LeftInBodyOfRight,
            Undefined,
            RightInBodyOfLeft,
            RightFirst,
            Count
        };

        enum class CacheStatus
        {
            Invalid = 0, //< Cache is empty
            Partial, //< Cache does not have all the orders between nodes
            Valid, //< Cache has all orders of nodes
            Count
        };

        /**
         * Return a full representation of 'n'
         */
        std::string   toString(NodeOrdering n);
        std::ostream& operator<<(std::ostream& stream, NodeOrdering n);

        /**
         * Return a full representation of 'c'
         */
        std::string   toString(CacheStatus c);
        std::ostream& operator<<(std::ostream& stream, CacheStatus c);

        /**
         * Return a 3-character representation of 'n'.
         */
        std::string abbrev(NodeOrdering n);

        /**
         * If ordering `order` applies to (a, b), return the ordering that applies to (b, a).
         */
        NodeOrdering opposite(NodeOrdering order);

        /**
         * Control flow graph.
         *
         * Nodes in the graph represent operations.  Edges describe
         * dependencies.
         */
        class ControlGraph : public Graph::Hypergraph<Operation, ControlEdge, false>
        {
        public:
            using Base = Graph::Hypergraph<Operation, ControlEdge, false>;

            ControlGraph() = default;

            /**
             * @brief Get a node/edge from the control graph.
             *
             * If the element specified by tag cannot be converted to
             * T, the return value is empty.
             *
             * @param tag Graph tag/index.
             */
            template <typename T>
            requires(std::constructible_from<ControlGraph::Element, T>) std::optional<T> get(
                int tag)
            const;

            /**
             * Returns the relative ordering of nodeA and nodeB according to the graph rules.
             *
             * From node X,
             *
             *  - Nodes connected via a Sequence edge (and their descendents) are after X.
             *  - Nodes connected via other kinds of edges (Initialize, Body, ForLoopIncrement)
             *    are in the body of X.
             *  - The descendents of X are ordered by the first kind of edge, in this order:
             *    Initialize -> Body -> ForLoopIncrement -> Sequence.
             *
             * Nodes whose relationship is not determined by the above rules could be
             * scheduled concurrently.
             */
            NodeOrdering compareNodes(int nodeA, int nodeB) const;

            /**
             * Yields (in no particular order) all nodes that are definitely after `node`.
             */
            Generator<int> nodesAfter(int node) const;

            /**
             * Yields (in no particular order) all nodes that are definitely before `node`.
             */
            Generator<int> nodesBefore(int node) const;

            /**
             * Yields (in no particular order) all nodes that are inside the body of `node`.
             * Note
             */
            Generator<int> nodesInBody(int node) const;

            /**
             * Yields (in no particular order) all nodes that contain `node` in their body.
             * Note
             */
            Generator<int> nodesContaining(int node) const;

            /**
             * Returns a string containing a text table describing the relationship between
             * all nodes in the graph.
             */
            std::string nodeOrderTableString() const;

            /**
             * Returns a string containing a text table describing the relationship between
             * the listed nodes in the graph.
             */
            std::string nodeOrderTableString(std::set<int> const& nodes) const;

            /**
             * Contains a map of every definite ordering between two nodes.
             * Note that node pairs whose ordering is undefined will be missing from the cache.
             * Also note that the entries are not duplicated for (nodeA, nodeB) and
             * (nodeB, nodeA). Only the entry where the first node ID is the lower number
             * will be present. The other entry can be obtained from opposite(nodeB, nodeA).
             *
             * Also, if a reference to the returned value is maintained through any changes
             * to the graph, the returned map will be cleared.
             */
            std::unordered_map<std::tuple<int, int>, NodeOrdering> const& nodeOrderTable() const;

            template <typename T>
            requires(std::constructible_from<Operation, T>)
                std::set<std::pair<int, int>> ambiguousNodes()
            const;

            /**
             * @brief Given two control stacks, add the necessary sequence edge such that the final nodes in the stack are relatively ordered.
             *
             * The final nodes in the control stacks are ordered by adding a sequence edge between the first nodes in the stacks that differ.
             *
             * If ordered is true, the order imposed is a -> b.
             * Otherwise, the order is inferred based on the following rules:
             * 1. If the differing nodes are translate time evaluatable setcoords, order the smaller one first.
             * 2. Otherwise, order the smaller numbered node first.
             *
             *
             * @tparam Range
             * @param aControlStack Path in the body-parent tree (computed from the control graph) from root to a.
             * @param bControlStack Path in the body-parent tree from root to b.
             * @param ordered Inputs are passed in order.
             */
            template <CForwardRangeOf<int> Range>
            void orderMemoryNodes(Range const& aControlStack,
                                  Range const& bControlStack,
                                  bool         ordered = true);

            template <typename T>
            inline std::predicate<int> auto isElemType() const
            {
                return [this](int x) -> bool { return get<T>(x).has_value(); };
            }

            /**
             * Connects each argument in order with a separate edge of type `Edge`. Accepts
             * any number of arguments.  The edges will be default-constructed.
             *
             * e.g. chain<Sequence>(a, b, c) will create two sequence edges, one from a to b,
             * and one from b to c.
             */
            template <CControlEdge Edge, std::convertible_to<int>... Nodes>
            void chain(int a, int b, Nodes... remaining);

        private:
            virtual void clearCache(Graph::GraphModification modification) override;
            void         checkOrderCache() const;
            void         populateOrderCache() const;

            /**
             * Populates m_orderCache for startingNodes relative to their descendents, and the
             * descendents of each relative to each other. Returns the descendents of
             * startingNodes.
             */
            template <CForwardRangeOf<int> Range>
            std::set<int> populateOrderCache(Range const& startingNodes) const;

            /**
             * Populates m_orderCache for startingNode relative to its descendents, and the
             * descendents relative to each other. Returns the descendents of startingNode.
             */
            std::set<int> populateOrderCache(int startingNode) const;

            NodeOrdering lookupOrderCache(int nodeA, int nodeB) const;
            void         writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const;

            template <CForwardRangeOf<int> ARange = std::initializer_list<int>,
                      CForwardRangeOf<int> BRange = std::initializer_list<int>>
            void writeOrderCache(ARange const& nodesA,
                                 BRange const& nodesB,
                                 NodeOrdering  order) const;

            mutable std::unordered_map<std::tuple<int, int>, NodeOrdering> m_orderCache;
            /**
             * If an entry is present, the value will be the IDs of every descendent from the key,
             * following every kind of edge.
             */
            mutable std::unordered_map<int, std::set<int>> m_descendentCache;

            mutable CacheStatus m_cacheStatus = CacheStatus::Invalid;
        };

        std::string name(ControlGraph::Element const& el);

        /**
         * @brief Determine if x holds an Operation of type T.
         */
        template <typename T>
        bool isOperation(auto const& x);

        /**
         * @brief Determine if x holds a ControlEdge of type T.
         */
        template <typename T>
        bool isEdge(auto const& x);
    }
}

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_impl.hpp>

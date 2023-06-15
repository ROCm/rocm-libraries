#pragma once

#include <variant>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Graph/Hypergraph_fwd.hpp>

#include "ControlEdge.hpp"
#include "KernelGraph/ControlGraph/Operation_fwd.hpp"
#include "Operation.hpp"

namespace rocRoller
{
    /**
     * Control flow routines.
     *
     * Control flow is represented as a graph.  Nodes in the control flow graph represent
     * operations (like load/store or a for loop).  Edges in the graph encode dependencies
     * between nodes.
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

        /**
         * Return a full representation of 'n'
         */
        std::string   toString(NodeOrdering n);
        std::ostream& operator<<(std::ostream& stream, NodeOrdering n);

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
            ControlGraph()
                : Graph::Hypergraph<Operation, ControlEdge, false>()
            {
            }

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
             * Returns a string containing a text table describing the relationship between
             * all nodes in the graph.
             */
            std::string nodeOrderTable() const;

        private:
            virtual void clearCache() override;
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

            /**
             * When non-empty, will contain a map of every definite ordering between two nodes.
             * Note that node pairs whose ordering is undefined will be missing from the cache.
             * Also note that the entries are not duplicated for (nodeA, nodeB) and
             * (nodeB, nodeA). Only the entry where the first node ID is the lower number
             * will be present. The other entry can be obtained from opposite(nodeB, nodeA).
             */
            mutable std::map<std::pair<int, int>, NodeOrdering> m_orderCache;
            /**
             * If an entry is present, the value will be the IDs of every descendent from the key,
             * following every kind of edge.
             */
            mutable std::map<int, std::set<int>> m_descendentCache;
        };

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

#include "ControlGraph_impl.hpp"

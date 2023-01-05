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
     * Control flow is represented as a graph.  Nodes in the
     * control flow graph represent operations (like load/store or a
     * for loop).  Edges in the graph encode dependencies between nodes.
     *
     */
    namespace KernelGraph::ControlGraph
    {
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

        private:
        };

        /**
         * @brief Determine if x holds an Operation of type T.
         */
        template <typename T>
        bool isOperation(auto x);

        /**
         * @brief Determine if x holds a ControlEdge of type T.
         */
        template <typename T>
        bool isEdge(auto x);

    }
}

#include "ControlGraph_impl.hpp"

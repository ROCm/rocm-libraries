#pragma once

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Graph/Hypergraph_fwd.hpp>

#include "ControlEdge.hpp"
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
    namespace KernelGraph::ControlHypergraph
    {
        /**
         * Control flow graph.
         *
         * Nodes in the graph represent operations.  Edges describe
         * dependencies.
         */
        class ControlHypergraph : public Graph::Hypergraph<Operation, ControlEdge, false>
        {
        public:
            ControlHypergraph()
                : Graph::Hypergraph<Operation, ControlEdge, false>()
            {
            }

        private:
        };
    }
}

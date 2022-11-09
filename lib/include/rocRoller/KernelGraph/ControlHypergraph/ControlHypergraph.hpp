#pragma once

#include <variant>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Graph/Hypergraph_fwd.hpp>

#include "ControlEdge.hpp"
#include "KernelGraph/ControlHypergraph/Operation_fwd.hpp"
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

            template <typename T>
            Generator<T> findOperations(int start) const
            {
                for(auto const x : depthFirstVisit(start))
                {
                    auto element = getElement(x);
                    if(std::holds_alternative<Operation>(element))
                    {
                        auto operation = std::get<Operation>(element);
                        if(std::holds_alternative<T>(operation))
                        {
                            co_yield std::get<T>(operation);
                        }
                    }
                }
            }

        private:
        };
    }
}

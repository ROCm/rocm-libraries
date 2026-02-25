// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphUtilities.hpp"

#include <concepts>
#include <vector>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace Graph
    {
        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        void removeRedundantEdges(AGraph& graph, EdgePredicate edgePredicate)
        {
            for(auto edge : findRedundantEdges(graph, edgePredicate))
            {
                graph.deleteElement(edge);
            }
        }

        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        Generator<int> findRedundantEdges(AGraph const& graph, EdgePredicate edgePredicate)
        {
            auto edges = graph.getEdges().filter(edgePredicate).template to<std::vector>();
            for(auto edge : edges)
            {
                auto tail = only(graph.template getNeighbours<Direction::Upstream>(edge)).value();
                auto head = only(graph.template getNeighbours<Direction::Downstream>(edge)).value();

                auto onlyFollowDifferentMatchingEdges
                    = [&](int x) -> bool { return x != edge && edgePredicate(x); };

                auto reachable = !graph
                                      .depthFirstVisit(tail,
                                                       onlyFollowDifferentMatchingEdges,
                                                       Direction::Downstream)
                                      .filter([head](int x) { return x == head; })
                                      .empty();

                if(reachable)
                {
                    co_yield edge;
                }
            }
        }
    }
}

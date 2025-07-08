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
                auto tail = graph.template getNeighbours<Direction::Upstream>(edge).only().value();
                auto head
                    = graph.template getNeighbours<Direction::Downstream>(edge).only().value();

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

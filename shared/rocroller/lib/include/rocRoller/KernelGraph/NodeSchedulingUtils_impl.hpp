
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

#include "NodeSchedulingUtils.hpp"

namespace rocRoller::KernelGraph::NodeScheduling
{
    std::unordered_map<int, std::vector<int>> getGroupedNodes(KernelGraph const&       graph,
                                                              std::predicate<int> auto pred)
    {
        auto theNodes = graph.control.getNodes().filter(pred);

        std::unordered_map<int, std::vector<int>> rv;
        for(auto node : theNodes)
        {
            auto parent = bodyParents(node, graph).take(1).only();
            AssertFatal(parent.has_value(), "Node has no body parent", ShowValue(node));

            rv[*parent].push_back(node);
        }
        return rv;
    }

    template <ControlGraph::COperation T>
    std::unordered_map<int, std::vector<int>> getGroupedNodes(KernelGraph const& graph)
    {
        auto pred = [&graph](int idx) { return graph.control.get<T>(idx).has_value(); };

        return getGroupedNodes(graph, pred);
    }

    void orderNodes(KernelGraph const& graph, std::vector<int>& nodes, auto const& comp)
    {
        std::vector<int> desiredOrder;

        if(Log::getLogger()->should_log(LogLevel::Debug))
        {
            std::set tmp(nodes.begin(), nodes.end());
            Log::debug("Pre-existing order:\n{}", graph.control.nodeOrderTableString(tmp));

            desiredOrder = nodes;
            std::ranges::sort(desiredOrder, comp);
        }
        // Simply including existing order in `BestNodeOrder` and calling `sort` can
        // lead to a situation where existing nodes appear out of program order.
        //
        // Instead:
        // 1. Create a subgraph that just contains `nodes` but preserves the same
        // order relationships between them.
        // 2. Walk that subgraph in topological order, using `BestNodeOrder` to decide
        // which node to pick next when there are multiple topologically valid options.

        auto subGraph = createSubGraph(graph, nodes);

        auto candidates = subGraph.roots().to<std::deque>();

        std::unordered_set<int> remainingNodes(nodes.begin(), nodes.end());
        remainingNodes.reserve(nodes.size());

        std::unordered_set<int> completedNodes;
        completedNodes.reserve(nodes.size());

        auto nodeSatisfied = [&](int node) -> bool {
            if(completedNodes.contains(node))
                return false;

            for(auto input : subGraph.getInputNodeIndices<ControlGraph::Sequence>(node))
            {
                if(!completedNodes.contains(input))
                    return false;
            }
            return true;
        };

        for(auto candidate : candidates)
        {
            remainingNodes.erase(candidate);
        }

        nodes.clear();

        if(Log::getLogger()->should_log(LogLevel::Debug))
        {
            std::ranges::sort(candidates, comp);
            Log::debug("Starting with ({})", fmt::join(candidates, ","));
        }

        while(!remainingNodes.empty() || !candidates.empty())
        {
            std::ranges::sort(candidates, comp);
            auto nextNode = candidates.front();
            candidates.pop_front();
            Log::debug("Picking {}", nextNode);

            nodes.push_back(nextNode);
            completedNodes.insert(nextNode);

            if(!remainingNodes.empty())
            {
                auto outputNodes = subGraph.getOutputNodeIndices<ControlGraph::Sequence>(nextNode);

                std::set<int> newNodes;

                for(auto outputNode : outputNodes)
                {
                    if(nodeSatisfied(outputNode))
                    {
                        newNodes.insert(outputNode);
                        candidates.push_back(outputNode);
                        remainingNodes.erase(outputNode);
                    }
                }

                if(!newNodes.empty() && Log::getLogger()->should_log(LogLevel::Debug))
                {
                    Log::debug("Adding ({})", fmt::join(newNodes, ", "));
                    std::ranges::sort(candidates, comp);
                    Log::debug("Now ({})", fmt::join(candidates, ", "));
                }
            }
        }

        Log::debug("Desired order: \n{}", fmt::join(desiredOrder, "\n"));
        Log::debug("Actual order: \n{}", fmt::join(nodes, "\n"));
    }
}
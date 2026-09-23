// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            auto parentPair = containingAncestors(node, graph).take(1).only();
            AssertFatal(parentPair.has_value(), "Node has no containing ancestor", ShowValue(node));

            rv[parentPair->first].push_back(node);
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
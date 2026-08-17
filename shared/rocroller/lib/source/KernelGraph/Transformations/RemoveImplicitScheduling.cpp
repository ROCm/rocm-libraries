// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/RemoveImplicitScheduling.hpp>

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace RemoveImplicitSchedulingDetail
    {
        void breakupNodes(KernelGraph& graph, std::vector<int> const& nodes)
        {
            auto getLoopOp = [&](int op) -> std::optional<int> {
                auto stack = controlStack(op, graph);

                for(auto parent : std::views::reverse(stack))
                {
                    if(graph.control.get<ControlGraph::ForLoopOp>(parent))
                        return parent;
                }

                return std::nullopt;
            };

            auto loop = getLoopOp(nodes.front()).value_or(-1);

            auto colouring = colourByUnrollValue(graph, -1);

            Log::debug(toString(colouring));

            using Colour = std::set<std::pair<int, int>>;

            std::map<Colour, std::vector<int>> reverse;

            for(auto node : nodes)
            {
                auto const& opColour = colouring.operationColour.at(node);

                Colour key(opColour.begin(), opColour.end());

                reverse[key].push_back(node);
            }

            for(auto& [key, keyOps] : reverse)
            {
                std::ranges::sort(keyOps, TopologicalCompare(graph));
            }

            std::set<int> edgesToKeep;

            for(auto& [key, keyOps] : reverse)
            {
                for(int idx = 0; idx + 1 < keyOps.size(); idx++)
                {
                    auto thisEdge = graph.control.findEdge(keyOps[idx], keyOps[idx + 1]);

                    if(!thisEdge)
                    {
                        AssertFatal(graph.control.compareNodes(
                                        UseCacheIfAvailable, keyOps[idx], keyOps[idx + 1])
                                    == ControlGraph::NodeOrdering::LeftFirst);

                        thisEdge = graph.control.addElement(
                            ControlGraph::Sequence(), {keyOps[idx]}, {keyOps[idx + 1]});
                    }

                    edgesToKeep.insert(*thisEdge);
                }
            }

            Log::debug("Keeping edges ({})", fmt::join(edgesToKeep, ", "));

            auto notMultiply = [&graph](int idx) {
                if(graph.control.getElementType(idx) != Graph::ElementType::Node)
                    return false;

                return !(graph.control.get<ControlGraph::Multiply>(idx).has_value());
            };

            std::map<int, int> connectionsToKeep;

            for(auto node : nodes)
            {
                auto upstreamNode
                    = graph.control.breadthFirstVisit(node, Graph::Direction::Upstream)
                          .filter(notMultiply)
                          .take(1)
                          .only();

                AssertFatal(upstreamNode.has_value(), ShowValue(node));

                AssertFatal(getLoopOp(node) == getLoopOp(*upstreamNode), ShowValue(node));

                connectionsToKeep[node] = *upstreamNode;
            }

            Log::debug("Got connections.");

            for(auto nodeA : nodes)
            {
                for(auto nodeB : nodes)
                {
                    if(nodeA == nodeB)
                        continue;

                    auto thisEdge = graph.control.findEdge(nodeA, nodeB);

                    if(thisEdge.has_value() && !edgesToKeep.contains(*thisEdge))
                    {
                        auto upstream = connectionsToKeep.at(nodeB);
                        auto order
                            = graph.control.compareNodes(UseCacheIfAvailable, upstream, nodeB);
                        AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                                        || order == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                                    ShowValue(order),
                                    ShowValue(upstream),
                                    ShowValue(nodeB),
                                    ShowValue(*thisEdge));
                        graph.control.deleteElement(*thisEdge);
                        if(order == ControlGraph::NodeOrdering::LeftFirst)
                            graph.control.chain<ControlGraph::Sequence>(upstream, nodeB);
                        else
                            graph.control.chain<ControlGraph::Body>(upstream, nodeB);
                    }
                }
            }
        }
    }

    KernelGraph RemoveImplicitScheduling::apply(KernelGraph const& original)
    {
        auto rv = original;

        auto groupedMultiplyNodes = NodeScheduling::getGroupedNodes<ControlGraph::Multiply>(rv);

        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            RemoveImplicitSchedulingDetail::breakupNodes(rv, nodes);
        }

        return rv;
    }
}

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

#include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes_detail.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace OrderMultiplyNodesDetail
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

        template <typename T>
        std::unordered_map<int, std::vector<int>> getGroupedNodes(KernelGraph const& graph)
        {
            auto pred = [&graph](int idx) { return graph.control.get<T>(idx).has_value(); };

            return getGroupedNodes(graph, pred);
        }

        std::unordered_map<int, std::vector<int>> getGroupedMultiplyNodes(KernelGraph const& graph)
        {
            return getGroupedNodes<ControlGraph::Multiply>(graph);
        }

        BestNodeOrder::BestNodeOrder(KernelGraph const& graph)
            : m_graph(graph)
            , m_tracer(graph)
        {
        }

        std::optional<bool> BestNodeOrder::existingOrder(int a, int b) const
        {
            if(a == b)
                return std::nullopt;

            auto existingOrder = m_graph.control.compareNodes(rocRoller::UpdateCache, a, b);

            if(existingOrder == ControlGraph::NodeOrdering::LeftFirst)
                return true;

            if(existingOrder == ControlGraph::NodeOrdering::RightFirst)
                return false;

            AssertFatal(existingOrder == ControlGraph::NodeOrdering::Undefined,
                        "These nodes should not contain each other",
                        ShowValue(a),
                        ShowValue(b));

            return std::nullopt;
        }

        std::vector<int> const& BestNodeOrder::aTagReplacements(int node) const
        {
            {
                auto iter = m_aTagReplacements.find(node);
                if(iter != m_aTagReplacements.end())
                    return iter->second;
            }

            auto getLoopOp = [&](int op) -> std::optional<int> {
                auto stack = controlStack(op, m_graph);

                for(auto parent : std::views::reverse(stack))
                {
                    if(m_graph.control.get<ControlGraph::ForLoopOp>(parent))
                        return parent;
                }

                return std::nullopt;
            };

            auto nodeLoop = getLoopOp(node);

            auto isLeft = [](ControlToCoordinateMapper::Connection const& c) {
                auto arg = getNaryArgument(c);
                return arg == NaryArgument::LHS || arg == NaryArgument::LHS_SCALE;
            };

            auto          connections_ = m_graph.mapper.getConnections(node);
            auto          connections  = connections_ | std::views::filter(isLeft);
            std::set<int> tags;

            for(auto const& connection : connections)
            {
                Log::debug("Op {}, Connection {}", node, toString(connection));
                tags.insert(connection.coordinate);
            }

            std::map<int, int> tagNodes;
            for(auto const& tag : tags)
            {
                auto tagTrace = m_tracer.coordinatesReadWrite(tag);

                for(auto const& rw : tagTrace)
                {
                    if(rw.rw == ControlFlowRWTracer::WRITE
                       && m_graph.control.get<ControlGraph::LoadLDSTile>(rw.control)
                       //    && m_graph.control.compareNodes(UpdateCache, node, rw.control)
                       //           == ControlGraph::NodeOrdering::LeftFirst
                       && getLoopOp(rw.control) == nodeLoop)
                    {
                        auto iter = tagNodes.find(tag);
                        if(iter == tagNodes.end()
                           || m_graph.control.compareNodes(UpdateCache, iter->second, rw.control)
                                  == ControlGraph::NodeOrdering::RightFirst)
                        {
                            tagNodes[tag] = rw.control;
                        }
                    }
                }
            }

            std::vector<int> rv;
            rv.reserve(tagNodes.size());

            for(auto const& [_, op] : tagNodes)
                rv.push_back(op);

            std::ranges::sort(rv, TopologicalCompare(m_graph));

            m_aTagReplacements[node] = std::move(rv);

            auto showEntry = [](std::pair<int, std::vector<int>> const& entry) {
                auto const& [tag, deps] = entry;
                return fmt::format("Multiply {}: ({})", tag, fmt::join(deps, ", "));
            };

            auto iter = m_aTagReplacements.find(node);
            Log::debug(showEntry(*iter));

            return m_aTagReplacements[node];
        }

        std::optional<bool> BestNodeOrder::orderByATagReplacements(int a, int b) const
        {
            auto const& as = aTagReplacements(a);
            auto const& bs = aTagReplacements(b);

            auto aIter = as.begin(), bIter = bs.begin();
            for(; aIter != as.end() && bIter != bs.end(); ++aIter, ++bIter)
            {
                if(auto order = existingOrder(*aIter, *bIter))
                    return *order;
            }

            if(aIter != as.end())
                return true;
            if(bIter != bs.end())
                return false;

            return std::nullopt;
        }

        std::optional<int> BestNodeOrder::downstreamMemoryNode(int node) const
        {
            auto iter = m_downstreamMemoryNodes.find(node);
            if(iter != m_downstreamMemoryNodes.end())
                return iter->second;

            auto isMemoryNode = [&](int idx) -> bool {
                auto node = m_graph.control.get<ControlGraph::Operation>(idx);
                if(!node.has_value())
                    return false;

                auto _isMemoryNode = []<typename T>(T const& theNode) {
                    using namespace ControlGraph;
                    return CIsAnyOf<T,
                                    LoadLDSTile,
                                    LoadLinear,
                                    LoadVGPR,
                                    LoadSGPR,
                                    LoadTiled,
                                    StoreLDSTile,
                                    LoadTileDirect2LDS,
                                    StoreLinear,
                                    StoreTiled,
                                    StoreVGPR,
                                    StoreSGPR>;
                };

                return std::visit(_isMemoryNode, *node);
            };

            auto downstreamMemoryNode
                = m_graph.control.breadthFirstVisit(node, Graph::Direction::Downstream)
                      .filter(isMemoryNode)
                      .take(1)
                      .only();

            m_downstreamMemoryNodes[node] = downstreamMemoryNode;
            return downstreamMemoryNode;
        }

        std::optional<bool> BestNodeOrder::orderByDownstreamMemoryNodes(int a, int b) const
        {
            auto downstreamMemoryNodeA = downstreamMemoryNode(a);
            auto downstreamMemoryNodeB = downstreamMemoryNode(b);

            if(downstreamMemoryNodeA.has_value() && downstreamMemoryNodeB.has_value())
            {
                return existingOrder(*downstreamMemoryNodeA, *downstreamMemoryNodeB);
            }

            if(downstreamMemoryNodeA.has_value())
                return true;

            if(downstreamMemoryNodeB.has_value())
                return false;

            return std::nullopt;
        }

        std::vector<int> const& BestNodeOrder::reversedTagDependencies(int node) const
        {
            auto iter = m_reversedTagDependencies.find(node);
            if(iter != m_reversedTagDependencies.end())
                return iter->second;

            auto          allRecords = m_tracer.coordinatesReadWrite();
            std::set<int> coordinatesReadByNode;
            for(auto const& rec : allRecords)
            {
                if(rec.control == node
                   && (rec.rw == ControlFlowRWTracer::READ
                       || rec.rw == ControlFlowRWTracer::READWRITE))
                    coordinatesReadByNode.insert(rec.coordinate);
            }

            std::vector<int> nodesThatWriteThoseCoordinatesBeforeTheNode;

            for(auto const& rec : allRecords)
            {
                if(rec.rw != ControlFlowRWTracer::READ && rec.control != node
                   && coordinatesReadByNode.contains(rec.coordinate)
                   && m_graph.control.compareNodes(UpdateCache, rec.control, node)
                          == ControlGraph::NodeOrdering::LeftFirst)
                {
                    nodesThatWriteThoseCoordinatesBeforeTheNode.push_back(rec.control);
                }
            }

            AssertFatal(!nodesThatWriteThoseCoordinatesBeforeTheNode.empty());

            auto reverseTopologicalCompare = [&](int a, int b) {
                return a != b
                       && m_graph.control.compareNodes(UpdateCache, a, b)
                              == ControlGraph::NodeOrdering::RightFirst;
            };

            std::sort(nodesThatWriteThoseCoordinatesBeforeTheNode.begin(),
                      nodesThatWriteThoseCoordinatesBeforeTheNode.end(),
                      reverseTopologicalCompare);

            m_reversedTagDependencies[node]
                = std::move(nodesThatWriteThoseCoordinatesBeforeTheNode);
            return m_reversedTagDependencies[node];
        }

        std::optional<bool> BestNodeOrder::orderByLastTagDependencies(int a, int b) const
        {
            auto const& as = reversedTagDependencies(a);
            auto const& bs = reversedTagDependencies(b);

            auto aIter = as.begin(), bIter = bs.begin();
            for(; aIter != as.end() && bIter != bs.end(); ++aIter, ++bIter)
            {
                if(auto order = existingOrder(*aIter, *bIter))
                    return *order;
            }

            if(aIter != as.end())
                return false;
            if(bIter != bs.end())
                return true;

            return std::nullopt;
        }

        bool BestNodeOrder::operator()(int a, int b) const
        {
            if(auto order = orderByATagReplacements(a, b))
                return *order;

            if(auto order = orderByDownstreamMemoryNodes(a, b))
                return *order;

            if(auto order = orderByLastTagDependencies(a, b))
                return *order;

            return a < b;
        }

        ControlGraph::ControlGraph createSubGraph(KernelGraph const&      graph,
                                                  std::vector<int> const& nodes)
        {
            ControlGraph::ControlGraph subGraph;

            for(auto node : nodes)
            {
                subGraph.setElement(node, graph.control.getElement(node));
            }

            for(auto iterA = nodes.begin(); iterA != nodes.end(); ++iterA)
            {
                for(auto iterB = iterA + 1; iterB != nodes.end(); ++iterB)
                {
                    auto order = graph.control.compareNodes(UpdateCache, *iterA, *iterB);

                    if(order == ControlGraph::NodeOrdering::LeftFirst)
                    {
                        subGraph.addElement(ControlGraph::Sequence{}, {*iterA}, {*iterB});
                    }
                    else if(order == ControlGraph::NodeOrdering::RightFirst)
                    {
                        subGraph.addElement(ControlGraph::Sequence{}, {*iterB}, {*iterA});
                    }
                    else
                    {
                        AssertFatal(order == ControlGraph::NodeOrdering::Undefined,
                                    "These nodes should not contain each other",
                                    ShowValue(*iterA),
                                    ShowValue(*iterB));
                    }
                }
            }

            removeRedundantSequenceEdges(subGraph);

            return subGraph;
        }

        ControlGraph::ControlGraph
            createOrderGraph(KernelGraph const& graph, std::vector<int> nodes, auto const& comp)
        {
            ControlGraph::ControlGraph rv;

            for(int idx : nodes)
                rv.setElement(idx, graph.control.getElement(idx));

            for(int idxA : nodes)
            {
                for(int idxB : nodes)
                {
                    if(idxA != idxB && comp(idxA, idxB))
                        rv.chain<ControlGraph::Sequence>(idxA, idxB);
                }
            }

            removeRedundantSequenceEdges(rv);

            return rv;
        }

        std::vector<int>
            getDesiredOrder(KernelGraph const& graph, std::vector<int> nodes, auto const& comp)
        {
            std::ranges::sort(nodes, comp);
            return nodes;
        }

        void BestNodeOrder::populateCache(auto range) const
        {
            for(auto x : range)
            {
                aTagReplacements(x);
                downstreamMemoryNode(x);
                reversedTagDependencies(x);
            }
        }

        void orderNodes(KernelGraph const& graph, std::vector<int>& nodes, auto const& comp)
        {
            {
                std::set tmp(nodes.begin(), nodes.end());
                Log::debug("Pre-existing order:\n{}", graph.control.nodeOrderTableString(tmp));
            }
            // Simply including existing order in `BestNodeOrder` and calling `sort` can
            // lead to a situation where existing nodes appear out of program order.
            //
            // Instead:
            // 1. Create a subgraph that just contains `nodes` but preserves the same
            // order relationships between them.
            // 2. Walk that subgraph in topological order, using `BestNodeOrder` to decide
            // which node to pick next when there are multiple topologically valid options.

            auto desiredOrder = getDesiredOrder(graph, nodes, comp);

            auto subGraph = OrderMultiplyNodesDetail::createSubGraph(graph, nodes);

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

            std::ranges::sort(candidates, comp);
            Log::debug("Starting with ({})", fmt::join(candidates, ","));

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
                    auto outputNodes
                        = subGraph.getOutputNodeIndices<ControlGraph::Sequence>(nextNode);

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

                    if(!newNodes.empty())
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

        struct ExchangeOrder
        {
            int getMultiply(int exchange) const
            {
                auto isMultiply = [this](int idx) {
                    return graph.control.get<ControlGraph::Multiply>(idx).has_value();
                };

                auto multiplies
                    = graph.control.getOutputNodeIndices<ControlGraph::Sequence>(exchange).filter(
                        isMultiply);

                auto iter = multiplies.begin();
                AssertFatal(iter != multiplies.end(), "No Multiply nodes attached to ", exchange);

                auto rv = *iter;

                ++iter;

                for(; iter != multiplies.end(); ++iter)
                {
                    auto candidate = *iter;

                    if(candidate != rv
                       && graph.control.compareNodes(UpdateCache, rv, candidate)
                              == ControlGraph::NodeOrdering::RightFirst)
                        rv = candidate;
                }

                return rv;
            }

            bool operator()(int a, int b) const
            {
                if(a == b)
                    return false;

                auto aMultiply = getMultiply(a);
                auto bMultiply = getMultiply(b);

                if(aMultiply == bMultiply)
                    return false;

                return TopologicalCompare(graph)(aMultiply, bMultiply);
            }

            KernelGraph const& graph;
        };

        void BestNodeOrder::logTagData() const
        {
            auto showEntry = [](std::pair<int, std::vector<int>> const& entry) {
                auto const& [tag, deps] = entry;
                return fmt::format("Multiply {}: ({})\n", tag, fmt::join(deps, ", "));
            };

            auto entries = m_aTagReplacements | std::views::transform(showEntry);

            Log::debug("Tag info: {}", fmt::join(entries, ""));
        }

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
                        graph.control.deleteElement(*thisEdge);
                        graph.control.chain<ControlGraph::Sequence>(connectionsToKeep.at(nodeB),
                                                                    nodeB);
                    }
                }
            }
        }

    }
    KernelGraph RemoveImplicitScheduling::apply(KernelGraph const& original)
    {
        auto rv = original;

        auto groupedMultiplyNodes = OrderMultiplyNodesDetail::getGroupedMultiplyNodes(rv);
        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            OrderMultiplyNodesDetail::breakupNodes(rv, nodes);
        }

        return rv;
    }

    KernelGraph OrderMultiplyNodes::apply(KernelGraph const& original)
    {
        auto rv = original;

        {
            auto groupedMultiplyNodes = OrderMultiplyNodesDetail::getGroupedMultiplyNodes(rv);
            for(auto& [parent, nodes] : groupedMultiplyNodes)
            {
                {
                    OrderMultiplyNodesDetail::BestNodeOrder comp(rv);
                    // Pre-populate cache because some STL algorithms take the comparator by value.
                    comp.populateCache(nodes);
                    OrderMultiplyNodesDetail::orderNodes(rv, nodes, comp);
                    comp.logTagData();
                }

                for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
                {
                    rv.control.chain<ControlGraph::Sequence>(nodes[idx], nodes[idx + 1]);
                }
            }
        }

        return rv;
    }

    KernelGraph OrderExchangeNodes::apply(KernelGraph const& original)
    {
        auto rv = original;

        {
            auto exchangeWithConnectedMultiply = [&rv](int node) {
                if(!rv.control.get<rocRoller::KernelGraph::ControlGraph::Exchange>(node)
                        .has_value())
                    return false;

                auto isMultiply = [&rv](int idx) -> bool {
                    return rv.control.get<rocRoller::KernelGraph::ControlGraph::Multiply>(idx)
                        .has_value();
                };

                auto hasOutputMultiplies
                    = rv.control
                          .getOutputNodeIndices<rocRoller::KernelGraph::ControlGraph::Sequence>(
                              node)
                          .filter(isMultiply)
                          .take(1)
                          .only()
                          .has_value();

                return hasOutputMultiplies;
            };

            auto groupedExchangeNodes
                = OrderMultiplyNodesDetail::getGroupedNodes(rv, exchangeWithConnectedMultiply);
            for(auto& [parent, nodes] : groupedExchangeNodes)
            {
                OrderMultiplyNodesDetail::ExchangeOrder comp{rv};
                OrderMultiplyNodesDetail::orderNodes(rv, nodes, comp);

                for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
                {
                    rv.control.chain<ControlGraph::Sequence>(nodes[idx], nodes[idx + 1]);

                    {
                        auto idxMultiply = comp.getMultiply(nodes[idx]);
                        if(rv.control.compareNodes(UseCacheIfAvailable, idxMultiply, nodes[idx + 1])
                           == ControlGraph::NodeOrdering::Undefined)
                        {
                            rv.control.chain<ControlGraph::Sequence>(idxMultiply, nodes[idx + 1]);
                        }
                    }

                    {
                        auto idxMultiply = comp.getMultiply(nodes[idx + 1]);
                        if(rv.control.compareNodes(UseCacheIfAvailable, idxMultiply, nodes[idx])
                           == ControlGraph::NodeOrdering::Undefined)
                        {
                            rv.control.chain<ControlGraph::Sequence>(nodes[idx], idxMultiply);
                        }
                    }
                }
            }
        }

        return rv;
    }

    ConstraintStatus NoUnorderedMultiplyNodes(const KernelGraph& k)
    {
        ConstraintStatus retval;

        auto groupedMultiplyNodes = OrderMultiplyNodesDetail::getGroupedMultiplyNodes(k);

        std::set<int> ambiguousNodes;

        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
            {
                if(k.control.compareNodes(UpdateCache, nodes[idx], nodes[idx + 1])
                   == ControlGraph::NodeOrdering::Undefined)
                {
                    ambiguousNodes.insert(nodes[idx]);
                    ambiguousNodes.insert(nodes[idx + 1]);
                }
            }
        }

        if(!ambiguousNodes.empty())
        {
            std::ostringstream msg;

            msg << "\\(";
            streamJoin(msg, ambiguousNodes, "|");
            msg << "\\)";

            retval.combine(false,
                           "Unordered multiply nodes found: " + ShowValue(ambiguousNodes)
                               + " Handy regex search string: " + msg.str());
        }

        return retval;
    }
}

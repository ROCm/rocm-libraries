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
        std::unordered_map<int, std::vector<int>> getGroupedMultiplyNodes(KernelGraph const& graph)
        {
            auto multiplyNodes = graph.control.getNodes().filter([&graph](int idx) {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            });

            std::unordered_map<int, std::vector<int>> rv;
            for(auto node : multiplyNodes)
            {
                auto parent = bodyParents(node, graph).take(1).only();
                AssertFatal(parent.has_value(), "Node has no body parent", ShowValue(node));

                rv[*parent].push_back(node);
            }
            return rv;
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

        void orderNodes(KernelGraph const& graph, std::vector<int>& nodes)
        {
            // Simply including existing order in `BestNodeOrder` and calling `sort` can
            // lead to a situation where existing nodes appear out of program order.
            //
            // Instead:
            // 1. Create a subgraph that just contains `nodes` but preserves the same
            // order relationships between them.
            // 2. Walk that subgraph in topological order, using `BestNodeOrder` to decide
            // which node to pick next when there are multiple topologically valid options.

            auto subGraph = OrderMultiplyNodesDetail::createSubGraph(graph, nodes);

            auto candidates = subGraph.roots().to<std::list>();

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

            BestNodeOrder comp(graph);

            while(!remainingNodes.empty() || !candidates.empty())
            {
                auto iter = candidates.begin();
                AssertFatal(iter != candidates.end());
                auto minIter = iter;
                ++iter;
                for(; iter != candidates.end(); ++iter)
                {
                    if(comp(*iter, *minIter))
                        minIter = iter;
                }

                auto nextNode = *minIter;
                candidates.erase(minIter);

                nodes.push_back(nextNode);
                completedNodes.insert(nextNode);

                if(!remainingNodes.empty())
                {
                    auto outputNodes
                        = subGraph.getOutputNodeIndices<ControlGraph::Sequence>(nextNode);

                    for(auto outputNode : outputNodes)
                    {
                        if(nodeSatisfied(outputNode))
                        {
                            candidates.push_back(outputNode);
                            remainingNodes.erase(outputNode);
                        }
                    }
                }
            }
        }
    }

    KernelGraph OrderMultiplyNodes::apply(KernelGraph const& original)
    {
        auto rv                   = original;
        auto groupedMultiplyNodes = OrderMultiplyNodesDetail::getGroupedMultiplyNodes(rv);
        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            OrderMultiplyNodesDetail::orderNodes(rv, nodes);

            for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
            {
                rv.control.chain<ControlGraph::Sequence>(nodes[idx], nodes[idx + 1]);
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

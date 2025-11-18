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

#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Graph/GraphUtilities.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

#define debug critical
#define Debug Critical

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace OrderMultiplyNodesDetail
        {

            std::string subGraphDOT(KernelGraph const& graph, std::vector<int> const& nodes)
            {
                ControlGraph::ControlGraph subGraph;

                std::map<int, int> nodeMap;
                ;

                for(auto node : nodes)
                {
                    nodeMap[node] = subGraph.addElement(graph.control.getElement(node));
                }

                for(auto iterA = nodes.begin(); iterA != nodes.end(); ++iterA)
                {
                    for(auto iterB = iterA + 1; iterB != nodes.end(); ++iterB)
                    {
                        auto order = graph.control.compareNodes(UpdateCache, *iterA, *iterB);

                        switch(order)
                        {
                        case ControlGraph::NodeOrdering::LeftFirst:
                            subGraph.chain<ControlGraph::Sequence>(nodeMap[*iterA],
                                                                   nodeMap[*iterB]);
                            break;
                        case ControlGraph::NodeOrdering::RightFirst:
                            subGraph.chain<ControlGraph::Sequence>(nodeMap[*iterB],
                                                                   nodeMap[*iterA]);
                            break;
                        case ControlGraph::NodeOrdering::LeftInBodyOfRight:
                            subGraph.chain<ControlGraph::Body>(nodeMap[*iterB], nodeMap[*iterA]);
                            break;
                        case ControlGraph::NodeOrdering::RightInBodyOfLeft:
                            subGraph.chain<ControlGraph::Body>(nodeMap[*iterA], nodeMap[*iterB]);
                            break;

                        case ControlGraph::NodeOrdering::Undefined:
                        case ControlGraph::NodeOrdering::Count:
                            break;
                        }
                    }
                }

                removeRedundantSequenceEdges(subGraph);
                removeRedundantBodyEdges(subGraph);

                return subGraph.toDOT();
            }

            std::unordered_map<int, std::vector<int>>
                getGroupedMultiplyNodes(KernelGraph const& graph)
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

            OrderByDownstreamMemoryNodes::OrderByDownstreamMemoryNodes(KernelGraph const& graph)
                : m_graph(graph)
            {
            }

            bool OrderByDownstreamMemoryNodes::operator()(int a, int b) const

            {
                auto downstreamMemoryNodeA = downstreamMemoryNode(a);
                auto downstreamMemoryNodeB = downstreamMemoryNode(b);

                if(downstreamMemoryNodeA.has_value() && downstreamMemoryNodeB.has_value())
                {
                    if(*downstreamMemoryNodeA == *downstreamMemoryNodeB)
                        return false;

                    auto order = m_graph.control.compareNodes(
                        UpdateCache, *downstreamMemoryNodeA, *downstreamMemoryNodeB);

                    if(order == ControlGraph::NodeOrdering::LeftFirst)
                        return true;
                    return false;
                }

                if(downstreamMemoryNodeA.has_value())
                    return true;

                if(downstreamMemoryNodeB.has_value())
                    return false;

                return false;
            }

            std::optional<int> OrderByDownstreamMemoryNodes::downstreamMemoryNode(int node) const
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

            OrderByLastTagDependencies::OrderByLastTagDependencies(KernelGraph const& graph)
                : m_graph(graph)
                , m_tracer(graph)
            {
            }

            bool OrderByLastTagDependencies::operator()(int a, int b) const
            {
                auto const& as = reversedTagDependencies(a);
                auto const& bs = reversedTagDependencies(b);

                auto aIter = as.begin(), bIter = bs.begin();
                for(; aIter != as.end() && bIter != bs.end(); ++aIter, ++bIter)
                {
                    if(*aIter == *bIter)
                        continue;

                    auto order = m_graph.control.compareNodes(UpdateCache, *aIter, *bIter);
                    if(order == ControlGraph::NodeOrdering::LeftFirst)
                        return true;
                    if(order == ControlGraph::NodeOrdering::RightFirst)
                        return false;
                    AssertFatal(order == ControlGraph::NodeOrdering::Undefined,
                                "These nodes should not contain each other",
                                ShowValue(*aIter),
                                ShowValue(*bIter));
                }

                if(aIter != as.end())
                    return false;
                if(bIter != bs.end())
                    return true;

                return false;
            }

            std::vector<int> const&
                OrderByLastTagDependencies::reversedTagDependencies(int node) const
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

            bool BestNodeOrder::operator()(int a, int b) const
            {
                // if(auto order = existingOrder(a, b))
                //     return *order;

                if(auto order = orderByDownstreamMemoryNodes(a, b))
                    return *order;

                if(auto order = orderByLastTagDependencies(a, b))
                    return *order;

                return a < b;
            }

            auto createSubGraph(KernelGraph const& graph, std::vector<int> const& nodes)
            {
                ControlGraph::ControlGraph subGraph;

                // std::map<int, int> nodeMap;

                for(auto node : nodes)
                {
                    // nodeMap[node] = subGraph.addElement(node);
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
                auto subGraph = OrderMultiplyNodesDetail::createSubGraph(graph, nodes);

                auto candidates = subGraph.roots().to<std::deque>();
                // candidates.reserve(nodes.size());

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
                std::ranges::sort(candidates, comp);

                std::vector<int> newCandidates;
                std::deque<int>  oldCandidates;

                while(!remainingNodes.empty())
                {
                    AssertFatal(!candidates.empty());

                    auto nextNode = candidates.front();
                    candidates.pop_front();
                    nodes.push_back(nextNode);
                    completedNodes.insert(nextNode);

                    auto outputNodes
                        = subGraph.getOutputNodeIndices<ControlGraph::Sequence>(nextNode);

                    for(auto outputNode : outputNodes)
                    {
                        if(nodeSatisfied(outputNode))
                        {
                            newCandidates.push_back(outputNode);
                            remainingNodes.erase(outputNode);
                        }
                    }

                    if(!newCandidates.empty())
                    {
                        std::ranges::sort(newCandidates, comp);

                        oldCandidates.swap(candidates);

                        std::ranges::merge(
                            oldCandidates, newCandidates, std::back_inserter(candidates), comp);

                        oldCandidates.clear();
                        newCandidates.clear();
                    }
                }

                nodes.insert(nodes.end(), candidates.begin(), candidates.end());
            }

            void orderNodes2(KernelGraph const& graph, std::vector<int>& nodes)
            {
                {
                    OrderByLastTagDependencies comp(graph);
                    std::ranges::stable_sort(nodes, comp);
                }

                {
                    OrderByDownstreamMemoryNodes comp(graph);
                    std::ranges::stable_sort(nodes, comp);
                }

                {
                    TopologicalCompare comp(graph);
                    std::ranges::stable_sort(nodes, comp);
                }
            }

        }

        KernelGraph OrderMultiplyNodes::apply(KernelGraph const& original)
        {
            auto rv                   = original;
            auto groupedMultiplyNodes = OrderMultiplyNodesDetail::getGroupedMultiplyNodes(rv);
            for(auto& [parent, nodes] : groupedMultiplyNodes)
            {
#if 1
                OrderMultiplyNodesDetail::orderNodes(rv, nodes);
#else
                std::ranges::sort(nodes, OrderMultiplyNodesDetail::BestNodeOrder(rv));

                for(auto iterA = nodes.begin(); iterA != nodes.end(); ++iterA)
                {
                    for(auto iterB = iterA + 1; iterB != nodes.end(); ++iterB)
                    {
                        if(rv.control.compareNodes(UpdateCache, *iterA, *iterB)
                           == ControlGraph::NodeOrdering::RightFirst)
                        {
                            Log::debug("{} <-> {}", *iterA, *iterB);
                        }
                    }
                }

// std::string dot = OrderMultiplyNodesDetail::subGraphDOT(rv, nodes);
// Log::debug("Ordering {}", dot);

// Log::debug("Order: {}", fmt::join(nodes, " -> "));
#endif

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
}

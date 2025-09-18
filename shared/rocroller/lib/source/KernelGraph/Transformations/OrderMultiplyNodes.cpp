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
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Graph/GraphUtilities.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::vector<std::set<int>>
            OrderMultiplyNodes::getLocallyUnorderedGroups(KernelGraph const& graph,
                                                          std::set<int>      nodes)
        {
            std::vector<std::set<int>> groupedNodes;

            std::unordered_map<int, std::optional<int>> containingForLoops;

            for(auto const& node : nodes)
            {
                containingForLoops[node]
                    = findContainingOperation<ControlGraph::ForLoopOp>(node, graph);
            }

            while(!nodes.empty())
            {
                std::set<int> group = {*nodes.begin()};
                nodes.erase(nodes.begin());

                auto setLoop = containingForLoops.at(*group.begin());

                for(auto node : nodes)
                {
                    auto nodeLoop = containingForLoops.at(node);

                    if(nodeLoop != setLoop)
                        continue;

                    bool canAdd = true;

                    for(auto existingNode : group)
                    {
                        if(graph.control.compareNodes(UpdateCache, existingNode, node)
                           != ControlGraph::NodeOrdering::Undefined)
                        {
                            canAdd = false;
                            break;
                        }
                    }

                    if(canAdd)
                        group.insert(node);
                }

                for(auto node : group)
                    nodes.erase(node);

                groupedNodes.push_back(std::move(group));
            }

            return groupedNodes;
        }

        std::vector<int>
            OrderMultiplyNodes::sortNodesByDownstreamMemoryNodes(KernelGraph const& graph,
                                                                 std::set<int>&     nodes)
        {
            auto isMemoryNode = [&](int idx) -> bool {
                auto node = graph.control.get<ControlGraph::Operation>(idx);
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

            auto downstreamMemoryNodes = [&]() {
                auto downstreamMemoryNodes = graph.control.depthFirstVisit(nodes)
                                                 .filter(isMemoryNode)
                                                 .to<std::unordered_set>();
                return std::vector(downstreamMemoryNodes.begin(), downstreamMemoryNodes.end());
            }();

            auto compare = TopologicalCompare(std::make_shared<KernelGraph>(graph));
            std::sort(downstreamMemoryNodes.begin(), downstreamMemoryNodes.end(), compare);

            Log::debug("Memory: {}", ShowValue(downstreamMemoryNodes));

            std::vector<int> orderedNodes;

            for(auto memNode : downstreamMemoryNodes)
            {
                for(auto upstreamNode :
                    graph.control.breadthFirstVisit(memNode, Graph::Direction::Upstream))
                {
                    if(nodes.contains(upstreamNode))
                    {
                        orderedNodes.push_back(upstreamNode);
                        nodes.erase(upstreamNode);
                        break;
                    }
                }

                if(nodes.empty())
                    break;
            }

            return orderedNodes;
        }

        std::vector<int> OrderMultiplyNodes::getReversedTagDependencies(
            KernelGraph const& graph, ControlFlowRWTracer const& tracer, int node)
        {
            auto          allRecords = tracer.coordinatesReadWrite();
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
                   && graph.control.compareNodes(UpdateCache, rec.control, node)
                          == ControlGraph::NodeOrdering::LeftFirst)
                {
                    nodesThatWriteThoseCoordinatesBeforeTheNode.push_back(rec.control);
                }
            }

            AssertFatal(!nodesThatWriteThoseCoordinatesBeforeTheNode.empty());

            auto comp = [&](int a, int b) {
                return a != b
                       && graph.control.compareNodes(UpdateCache, a, b)
                              == ControlGraph::NodeOrdering::RightFirst;
            };

            std::sort(nodesThatWriteThoseCoordinatesBeforeTheNode.begin(),
                      nodesThatWriteThoseCoordinatesBeforeTheNode.end(),
                      comp);

            return nodesThatWriteThoseCoordinatesBeforeTheNode;
        }

        std::vector<int>
            OrderMultiplyNodes::sortNodesByLastTagDependencies(KernelGraph const&   graph,
                                                               std::set<int> const& nodes)
        {
            ControlFlowRWTracer tracer(graph);

            using NodeWithDeps = std::tuple<std::vector<int>, int>;
            std::vector<NodeWithDeps> nodesWithUpstreamDependencies;
            for(auto node : nodes)
            {
                nodesWithUpstreamDependencies.emplace_back(
                    getReversedTagDependencies(graph, tracer, node), node);
            }

            auto compareNodeVectors = [&](NodeWithDeps const& ap, NodeWithDeps const& bp) {
                auto comp = [&](int a, int b) {
                    return a != b
                           && graph.control.compareNodes(UpdateCache, a, b)
                                  == ControlGraph::NodeOrdering::LeftFirst;
                };
                auto const& as = std::get<0>(ap);
                auto const& bs = std::get<0>(bp);

                return std::lexicographical_compare(
                    as.begin(), as.end(), bs.begin(), bs.end(), comp);
            };

            std::sort(nodesWithUpstreamDependencies.begin(),
                      nodesWithUpstreamDependencies.end(),
                      compareNodeVectors);

            for(auto const& [ups, node] : nodesWithUpstreamDependencies)
                Log::debug("{}: {}", node, ups);

            std::vector<int> rv;
            for(auto const& [_, node] : nodesWithUpstreamDependencies)
                rv.push_back(node);
            return rv;
        }

        std::vector<std::vector<int>>
            OrderMultiplyNodes::findAndOrderGroups(KernelGraph const& graph)
        {
            auto isMultiply = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            };

            auto allMultiplyNodes = graph.control.getNodes().filter(isMultiply).to<std::set>();

            auto groupedNodes = getLocallyUnorderedGroups(graph, std::move(allMultiplyNodes));

            std::vector<std::vector<int>> groupedOrderedNodes;

            std::vector<std::vector<std::vector<ControlToCoordinateMapper::Connection>>>
                multiplyConnections;

            for(auto& group : groupedNodes)
            {
                Log::debug("Group of nodes: {}", ShowValue(group));

                auto orderedNodes = sortNodesByDownstreamMemoryNodes(graph, group);

                Log::debug("Nodes ordered by memory: {}, remaining: {}",
                           ShowValue(orderedNodes),
                           ShowValue(group));

                if(!(orderedNodes.empty() xor group.empty()))
                {
                    group.insert(orderedNodes.begin(), orderedNodes.end());
                }

                if(!group.empty())
                {
                    orderedNodes = sortNodesByLastTagDependencies(graph, group);

                    Log::debug("Nodes ordered by upstream dependencies: {}",
                               ShowValue(orderedNodes));
                }

                groupedOrderedNodes.push_back(std::move(orderedNodes));
            }

            return groupedOrderedNodes;
        }

        KernelGraph OrderMultiplyNodes::apply(KernelGraph const& original)
        {
            auto rv = original;

            for(auto const& group : findAndOrderGroups(original))
            {
                for(size_t idx = 0; idx + 1 < group.size(); idx++)
                {
                    rv.control.chain<ControlGraph::Sequence>(group[idx], group[idx + 1]);
                }
            }

            return rv;
        }
    }
}

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        /**
         * @brief A struct to record a loop's parent loops (the loops that
         *        contain this loop inside their Body) and child loops (the
         *        loops that are inside the Body of this loop)
         */
        struct LoopBodyInfo
        {
            std::unordered_set<int> parentLoops;
            std::set<int>           childLoops;
        };

        /**
         * @brief Fuse Loops Transformation
         *
         * This transformation looks for the following pattern:
         *
         *     ForLoop
         * |            |
         * Set Coord    Set Coord
         * |            |
         * For Loop     For Loop
         *
         * If it finds it, it will fuse the lower loops into a single
         * loop, as long as they are the same size.
         */
        namespace FuseLoopsNS
        {
            using GD = rocRoller::Graph::Direction;

            /**
             * @brief Find a path from a node to a ForLoopOp using only Sequence edges
             *
             * Returns an empty vector if no path is found.
             *
             * @param graph
             * @param start
             * @return std::vector<int>
             */
            std::vector<int> pathToForLoop(KernelGraph& graph, int start)
            {
                // Find the first ForLoop under the node
                auto allForLoops
                    = graph.control
                          .findNodes(
                              start,
                              [&](int tag) -> bool {
                                  return isOperation<ForLoopOp>(graph.control.getElement(tag));
                              },
                              GD::Downstream)
                          .to<std::vector>();

                if(allForLoops.empty())
                    return {};

                auto firstForLoop = allForLoops[0];

                // Find all of the nodes in between the node and the first for loop
                auto pathToLoopWithEdges = graph.control
                                               .path<GD::Downstream>(std::vector<int>{start},
                                                                     std::vector<int>{firstForLoop})
                                               .to<std::vector>();

                // Filter out only the nodes
                std::vector<int> pathToLoop;
                std::copy_if(pathToLoopWithEdges.begin(),
                             pathToLoopWithEdges.end(),
                             std::back_inserter(pathToLoop),
                             [&](int tag) -> bool {
                                 return graph.control.getElementType(tag)
                                        == Graph::ElementType::Node;
                             });

                return pathToLoop;
            }

            /**
             * @brief Order a new group (nodes) with existing groups. Each group consists of
             *        a pair of nodes which are the first and last nodes of memory nodes in a
             *        for-loop.
             *
             */
            static void orderGroups(rocRoller::KernelGraph::KernelGraph& graph,
                                    std::set<std::pair<int, int>>&       groups,
                                    std::vector<int>&                    nodes)

            {
                if(nodes.empty())
                    return;

                //
                // Create a new group using the first and last node of `nodes`.
                // An assumption here is `nodes` should be totally ordered.
                //
                auto [firstNode, lastNode] = getFirstAndLastNodes(graph, nodes);
                if(groups.empty())
                {
                    groups.emplace(firstNode, lastNode);
                    return;
                }

                auto [new_group, inserted] = groups.emplace(firstNode, lastNode); // Add new group
                AssertFatal(inserted); // Should not have identical group

                //
                // Order (insert sequence edges) the new group with existing groups.
                // An important assumption here is groups should not overlap.
                //
                if(new_group != groups.begin())
                {
                    auto prev_group = std::prev(new_group);
                    AssertFatal(prev_group->second < firstNode);
                    auto sc1 = getTopSetCoordinate(graph, prev_group->second);
                    auto sc2 = getTopSetCoordinate(graph, firstNode);
                    graph.control.addElement(Sequence(), {sc1}, {sc2});
                }

                if(*new_group != *groups.rbegin())
                {
                    auto next_group = std::next(new_group);
                    AssertFatal(next_group->first > lastNode);
                    auto sc1 = getTopSetCoordinate(graph, lastNode);
                    auto sc2 = getTopSetCoordinate(graph, next_group->first);
                    graph.control.addElement(Sequence(), {sc1}, {sc2});
                }
            }

            void fuseLoops(KernelGraph&                           graph,
                           int                                    tag,
                           std::unordered_map<int, LoopBodyInfo>& loopInfo)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops({})", tag);

                //
                // Return early if no sufficient loops to fuse
                //
                if(loopInfo.at(tag).childLoops.size() < 2)
                    return;

                auto dontWalkPastForLoop = [&](int tag) -> bool {
                    for(auto neighbour : graph.control.getNeighbours(tag, GD::Downstream))
                    {
                        if(graph.control.get<ForLoopOp>(neighbour))
                        {
                            return false;
                        }
                    }
                    return true;
                };

                // See if any of the ForLoopOps that were found in paths
                // should be fused together.
                std::unordered_set<int>   forLoopsToFuse;
                Expression::ExpressionPtr loopIncrement;
                Expression::ExpressionPtr loopLength;
                for(auto const& forLoop : loopInfo.at(tag).childLoops)
                {
                    if(forLoopsToFuse.count(forLoop) != 0)
                        return;

                    // Check to see if loops are all the same length
                    auto forLoopDim = getSize(std::get<CT::Dimension>(graph.coordinates.getElement(
                        graph.mapper.get(forLoop, NaryArgument::DEST))));
                    if(loopLength)
                    {
                        if(!identical(forLoopDim, loopLength))
                            return;
                    }
                    else
                    {
                        loopLength = forLoopDim;
                    }

                    // Check to see if loops are incremented by the same value
                    auto [dataTag, increment] = getForLoopIncrement(graph, forLoop);
                    if(loopIncrement)
                    {
                        if(!identical(loopIncrement, increment))
                            return;
                    }
                    else
                    {
                        loopIncrement = increment;
                    }

                    forLoopsToFuse.insert(forLoop);
                }

                if(forLoopsToFuse.size() <= 1)
                    return;

                auto fusedLoopTag = *forLoopsToFuse.begin();

                auto fusedLoopBodyChildren
                    = graph.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();

                auto initializeGroups = [&]<typename T>() {
                    std::set<std::pair<int, int>> groups;
                    auto                          nodes
                        = filter(graph.control.isElemType<T>(),
                                 graph.control.depthFirstVisit(
                                     fusedLoopBodyChildren, dontWalkPastForLoop, GD::Downstream))
                              .template to<std::vector>();
                    if(not nodes.empty())
                        groups.emplace(getFirstAndLastNodes(graph, nodes));
                    return groups;
                };

                auto groups_loads     = initializeGroups.template operator()<LoadTiled>();
                auto groups_ldsLoads  = initializeGroups.template operator()<LoadLDSTile>();
                auto groups_stores    = initializeGroups.template operator()<StoreTiled>();
                auto groups_ldsStores = initializeGroups.template operator()<StoreLDSTile>();

                for(auto const& forLoopTag : forLoopsToFuse)
                {
                    if(forLoopTag == fusedLoopTag)
                        continue;

                    for(auto const& child :
                        graph.control.getOutputNodeIndices<Sequence>(forLoopTag).to<std::vector>())
                    {
                        if(fusedLoopTag != child)
                        {
                            graph.control.addElement(Sequence(), {fusedLoopTag}, {child});
                        }
                        graph.control.deleteElement<Sequence>(std::vector<int>{forLoopTag},
                                                              std::vector<int>{child});
                        std::unordered_set<int> toDelete;
                        for(auto descSeqOfChild :
                            filter(graph.control.isElemType<Sequence>(),
                                   graph.control.depthFirstVisit(child, GD::Downstream)))
                        {
                            if(graph.control.getNeighbours<GD::Downstream>(descSeqOfChild)
                                   .to<std::unordered_set>()
                                   .contains(fusedLoopTag))
                            {
                                toDelete.insert(descSeqOfChild);
                            }
                        }
                        for(auto edge : toDelete)
                        {
                            graph.control.deleteElement(edge);
                        }
                    }

                    //
                    // Extract the memory nodes in forLoopTag, which will be used
                    // at the end to order with memory nodes in fusedLoopTag.
                    //
                    std::vector<int> loads;
                    std::vector<int> ldsLoads;
                    std::vector<int> stores;
                    std::vector<int> ldsStores;
                    {
                        auto children = graph.control.getOutputNodeIndices<Body>(forLoopTag)
                                            .to<std::vector>();

                        loads = filter(graph.control.isElemType<LoadTiled>(),
                                       graph.control.depthFirstVisit(
                                           children, dontWalkPastForLoop, GD::Downstream))
                                    .to<std::vector>();
                        ldsLoads = filter(graph.control.isElemType<LoadLDSTile>(),
                                          graph.control.depthFirstVisit(
                                              children, dontWalkPastForLoop, GD::Downstream))
                                       .to<std::vector>();
                        stores = filter(graph.control.isElemType<StoreTiled>(),
                                        graph.control.depthFirstVisit(
                                            children, dontWalkPastForLoop, GD::Downstream))
                                     .to<std::vector>();
                        ldsStores = filter(graph.control.isElemType<StoreLDSTile>(),
                                           graph.control.depthFirstVisit(
                                               children, dontWalkPastForLoop, GD::Downstream))
                                        .to<std::vector>();
                    }

                    for(auto const& child :
                        graph.control.getOutputNodeIndices<Body>(forLoopTag).to<std::vector>())
                    {
                        graph.control.addElement(Body(), {fusedLoopTag}, {child});
                        graph.control.deleteElement<Body>(std::vector<int>{forLoopTag},
                                                          std::vector<int>{child});
                    }

                    //
                    // Set the children loops of forLoopTag to be the children of fusedLoopTag
                    //
                    for(auto& child : loopInfo.at(forLoopTag).childLoops)
                    {
                        if(loopInfo.contains(child))
                        {
                            loopInfo.at(child).parentLoops.insert(fusedLoopTag);
                            loopInfo.at(fusedLoopTag).childLoops.insert(child);
                            loopInfo.at(child).parentLoops.erase(forLoopTag);
                        }
                    }

                    for(auto const& parent :
                        graph.control.getInputNodeIndices<Sequence>(forLoopTag).to<std::vector>())
                    {
                        auto descOfFusedLoop
                            = graph.control
                                  .depthFirstVisit(fusedLoopTag,
                                                   graph.control.isElemType<Sequence>(),
                                                   GD::Downstream)
                                  .to<std::unordered_set>();

                        if(!descOfFusedLoop.contains(parent))
                        {
                            graph.control.addElement(Sequence(), {parent}, {fusedLoopTag});
                        }
                        graph.control.deleteElement<Sequence>(std::vector<int>{parent},
                                                              std::vector<int>{forLoopTag});
                    }

                    for(auto const& parent :
                        graph.control.getInputNodeIndices<Body>(forLoopTag).to<std::vector>())
                    {
                        graph.control.addElement(Body(), {parent}, {fusedLoopTag});
                        graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                          std::vector<int>{forLoopTag});
                    }

                    //
                    // Set the parent loops of forLoopTag to be the parent of fusedLoopTag
                    // and remove forLoopTag from loopInfo.
                    //
                    for(auto parent : loopInfo.at(forLoopTag).parentLoops)
                    {
                        loopInfo.at(parent).childLoops.insert(fusedLoopTag);
                        loopInfo.at(fusedLoopTag).parentLoops.insert(parent);
                    }
                    loopInfo.erase(forLoopTag);

                    purgeFor(graph, forLoopTag);

                    //
                    // Order the memory nodes in forLoopTag with memory
                    // nodes in fusedLoopTag.
                    //
                    // An important assumption here is the memory nodes of
                    // forLoopTag should be ordered totally already, and
                    // orderGroups leverages this fact to connect the first and
                    // last nodes with other groups to achieve total ordering.
                    //
                    orderGroups(graph, groups_loads, loads);
                    orderGroups(graph, groups_ldsLoads, ldsLoads);
                    orderGroups(graph, groups_stores, stores);
                    orderGroups(graph, groups_ldsStores, ldsStores);
                }
            }
        }

        static std::optional<int>
            getChildLoopInsideBody(KernelGraph const& kg, int tag, std::unordered_set<int>& visited)
        {
            if(isOperation<ForLoopOp>(kg.control.getElement(tag)))
                return tag;

            std::optional<int> ret;
            for(auto node : kg.control.getOutputNodeIndices(tag, [](ControlEdge) { return true; }))
            {
                if(visited.contains(node))
                    continue;

                visited.insert(node);
                auto childLoop = getChildLoopInsideBody(kg, node, visited);
                if(childLoop.has_value())
                {
                    //
                    // We can early return here. But to verify the assumption
                    // that each Body edge has one loop at most, we continue
                    // the traversal.
                    //
                    AssertFatal(not ret.has_value(),
                                "Each Body edge should contain at most only one loop");
                    ret = childLoop;
                }
            }
            return ret;
        }

        static void populateChildLoops(KernelGraph const&                     kg,
                                       int                                    tag,
                                       std::unordered_map<int, LoopBodyInfo>& loopInfo)
        {
            std::unordered_set<int> visited;
            loopInfo[tag];

            for(auto node : kg.control.getOutputNodeIndices<Body>(tag))
            {
                visited.clear();
                auto childLoop = getChildLoopInsideBody(kg, node, visited);
                if(childLoop.has_value())
                    loopInfo.at(tag).childLoops.insert(childLoop.value());
            }
        }

        KernelGraph FuseLoops::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");

            auto newGraph = k;

            std::unordered_map<int, LoopBodyInfo> loopInfo;
            {
                auto loops = newGraph.control.getNodes<ForLoopOp>().to<std::vector>();
                for(auto loop : loops)
                    populateChildLoops(newGraph, loop, loopInfo);

                //
                // Populate parent loops
                //
                for(auto& [loop, info] : loopInfo)
                    for(auto& child : info.childLoops)
                        loopInfo.at(child).parentLoops.insert(loop);
            }

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    FuseLoopsNS::fuseLoops(newGraph, node, loopInfo);
                }
            }

            return newGraph;
        }
    }
}

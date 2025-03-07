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

            void fuseLoops(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops({})", tag);

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

                // Find all of the paths from the top of one of a body to a
                // ForLoopOp.
                auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                std::vector<std::vector<int>> paths;
                for(auto const& body : bodies)
                {
                    auto path = pathToForLoop(graph, body);
                    if(!path.empty())
                        paths.push_back(path);
                }

                // See if any of the ForLoopOps that were found in paths
                // should be fused together.
                std::unordered_set<int>   forLoopsToFuse;
                Expression::ExpressionPtr loopIncrement;
                Expression::ExpressionPtr loopLength;
                for(auto const& path : paths)
                {
                    auto forLoop = path.back();
                    if(forLoopsToFuse.count(forLoop) != 0)
                        return;

                    // Check to see if loops are all the same length
                    auto dimTag     = graph.mapper.get(forLoop, NaryArgument::DEST);
                    auto forLoopDim = getSize(graph.coordinates.getNode(dimTag));
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

                    for(auto const& child :
                        graph.control.getOutputNodeIndices<Body>(forLoopTag).to<std::vector>())
                    {
                        graph.control.addElement(Body(), {fusedLoopTag}, {child});
                        graph.control.deleteElement<Body>(std::vector<int>{forLoopTag},
                                                          std::vector<int>{child});
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

                    purgeFor(graph, forLoopTag);
                }

                auto children
                    = graph.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();

                auto loads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(
                                        children, dontWalkPastForLoop, GD::Downstream))
                                 .to<std::vector>();

                auto ldsLoads = filter(graph.control.isElemType<LoadLDSTile>(),
                                       graph.control.depthFirstVisit(
                                           children, dontWalkPastForLoop, GD::Downstream))
                                    .to<std::vector>();

                auto stores = filter(graph.control.isElemType<StoreTiled>(),
                                     graph.control.depthFirstVisit(
                                         children, dontWalkPastForLoop, GD::Downstream))
                                  .to<std::vector>();

                auto ldsStores = filter(graph.control.isElemType<StoreLDSTile>(),
                                        graph.control.depthFirstVisit(
                                            children, dontWalkPastForLoop, GD::Downstream))
                                     .to<std::vector>();

                orderMemoryNodes(graph, loads, true);
                orderMemoryNodes(graph, ldsLoads, true);
                orderMemoryNodes(graph, stores, true);
                orderMemoryNodes(graph, ldsStores, true);
            }
        }

        KernelGraph FuseLoops::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");

            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    FuseLoopsNS::fuseLoops(newGraph, node);
                }
            }

            return newGraph;
        }
    }
}

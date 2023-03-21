#include <rocRoller/KernelGraph/KernelGraph.hpp>
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
        namespace FuseLoops
        {
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
                              Graph::Direction::Downstream)
                          .to<std::vector>();

                if(allForLoops.empty())
                    return {};

                auto firstForLoop = allForLoops[0];

                // Find all of the nodes in between the node and the first for loop
                auto pathToLoopWithEdges
                    = graph.control
                          .path<Graph::Direction::Downstream>(std::vector<int>{start},
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
                auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();

                // Find all of the paths from the top of one of a body to a
                // ForLoopOp.
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
                    auto forLoopDim = getSize(std::get<Dimension>(
                        graph.coordinates.getElement(graph.mapper.get<Dimension>(forLoop))));
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
                        graph.control.addElement(Sequence(), {fusedLoopTag}, {child});
                        graph.control.deleteElement<Sequence>(std::vector<int>{forLoopTag},
                                                              std::vector<int>{child});
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
                        graph.control.addElement(Sequence(), {parent}, {fusedLoopTag});
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
            }
        }

        KernelGraph fuseLoops(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");
            rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops()");

            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    FuseLoops::fuseLoops(newGraph, node);
                }
            }

            return newGraph;
        }
    }
}

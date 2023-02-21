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

            /**
             * @brief Insert Set Coordinate nodes underneath fused loop
             *
             * @tparam EdgeType
             * @param graph
             * @param path
             * @param origSetCoord
             * @param fusedLoopTag
             */
            template <CControlEdge EdgeType>
            std::vector<int> insertSetCoordinates(KernelGraph&            graph,
                                                  std::vector<int> const& path,
                                                  int                     origSetCoord,
                                                  int                     fusedLoopTag)
            {
                auto forLoopTag = path.back();

                auto children = graph.control.getOutputNodeIndices<EdgeType>(forLoopTag)
                                    .template to<std::vector>();
                if(children.empty())
                    return {};

                // Add the top SetCoord node
                auto setCoord = graph.control.addElement(graph.control.getElement(origSetCoord));
                graph.control.addElement(EdgeType(), {fusedLoopTag}, {setCoord});
                for(auto const& c : graph.mapper.getConnections(origSetCoord))
                {
                    graph.mapper.connect(setCoord, c.coordinate, c.connection);
                }

                // Find any other SetCoord nodes within the path, and add them after the
                // top one
                for(auto const& node : path)
                {
                    if(isOperation<SetCoordinate>(graph.control.getElement(node)))
                    {
                        auto prevSetCoord = setCoord;
                        setCoord = graph.control.addElement(graph.control.getElement(node));
                        graph.control.addElement(Body(), {prevSetCoord}, {setCoord});
                        for(auto const& c : graph.mapper.getConnections(node))
                        {
                            graph.mapper.connect(setCoord, c.coordinate, c.connection);
                        }
                    }
                }

                for(auto const& child : children)
                {
                    graph.control.addElement(Body(), {setCoord}, {child});
                    graph.control.deleteElement<EdgeType>(std::vector<int>{forLoopTag},
                                                          std::vector<int>{child});
                }

                return children;
            }

            void fuseLoops(KernelGraph& graph, int tag)
            {
                auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();

                // Find all of the SetCoordinate nodes that are at the top of
                // one of the ForLoopOp bodies
                std::vector<int> setCoords;
                for(auto const& body : bodies)
                {
                    if(isOperation<SetCoordinate>(graph.control.getElement(body)))
                        setCoords.push_back(body);
                }

                if(setCoords.empty())
                    return;

                // Find all of the paths from one of the SetCoordinate nodes to a
                // ForLoopOp.
                // paths is a map whose key is one of the SetCoordinate nodes
                // and whose value is a vector containing all of the paths to
                // ForLoopOps.
                std::map<int, std::vector<std::vector<int>>> paths;
                for(auto const& setCoord : setCoords)
                {
                    auto setCoordBodies
                        = graph.control.getOutputNodeIndices<Body>(setCoord).to<std::set>();
                    std::vector<std::vector<int>> setCoordPaths;
                    for(auto const& body : setCoordBodies)
                    {
                        auto path = pathToForLoop(graph, body);
                        if(!path.empty())
                            setCoordPaths.push_back(path);
                    }

                    paths[setCoord] = setCoordPaths;
                }

                // See if any of the ForLoopOps that were found in paths
                // should be fused together.
                std::unordered_set<int>   forLoopsToFuse;
                Expression::ExpressionPtr loopIncrement;
                Expression::ExpressionPtr loopLength;
                for(auto const& setCoordPaths : paths)
                {
                    for(auto const& path : setCoordPaths.second)
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
                }

                if(forLoopsToFuse.size() <= 1)
                    return;

                // Insert SetCoord between each Body and Sequence Node
                auto coordPath    = paths.begin();
                auto origSetCoord = coordPath->first;
                auto forLoopTag   = coordPath->second[0].back();

                // The first ForLoop will be the one that all of the others are fused into
                auto fusedLoopTag = forLoopTag;

                // Iterate through the rest of the for loops to be fused
                for(; coordPath != paths.end(); ++coordPath)
                {
                    auto             origSetCoord = coordPath->first;
                    std::vector<int> allSequenceNodes;
                    for(auto const& path : coordPath->second)
                    {
                        auto forLoopTag = path.back();

                        // For each node (N) leaving for loop with Body or Sequence edge:
                        // Delete edge from original for loop to N
                        // Add edge from new for loop to SetCoord. Add edge from SetCoord to N
                        auto sequenceNodes = insertSetCoordinates<Sequence>(
                            graph, path, origSetCoord, fusedLoopTag);
                        allSequenceNodes.insert(
                            allSequenceNodes.end(), sequenceNodes.begin(), sequenceNodes.end());
                        insertSetCoordinates<Body>(graph, path, origSetCoord, fusedLoopTag);

                        // For each parent (N) of for loop
                        // Delete edge from N to original for loop
                        auto pathSize = path.size();
                        if(pathSize > 1)
                        {
                            auto N = path[pathSize - 2];
                            graph.control.deleteElement<Sequence>(std::vector<int>{N},
                                                                  std::vector<int>{forLoopTag});
                        }
                        else
                        {
                            graph.control.deleteElement<Body>(std::vector<int>{coordPath->first},
                                                              std::vector<int>{forLoopTag});
                        }

                        if(fusedLoopTag != forLoopTag)
                        {
                            // Delete old For loop, as well as its initialize and increment nodes.
                            auto forLoopChildren
                                = graph.control.depthFirstVisit(forLoopTag).to<std::vector>();
                            for(auto const& toDelete : forLoopChildren)
                            {
                                graph.control.deleteElement(toDelete);
                            }
                        }
                    }

                    // Delete all Sequence edges from children of SetCoord to children of fusedLoopTag
                    auto forLoopSequenceChildren
                        = graph.control.depthFirstVisit(allSequenceNodes).to<std::set>();
                    auto setCoordChildren
                        = graph.control.depthFirstVisit(coordPath->first).to<std::set>();

                    for(auto const& setCoordChild : setCoordChildren)
                    {
                        if(forLoopSequenceChildren.count(setCoordChild) == 1)
                        {
                            auto location = graph.control.getLocation(setCoordChild);
                            for(auto const& incomingEdge : location.incoming)
                            {
                                auto parents
                                    = graph.control
                                          .getNeighbours<Graph::Direction::Upstream>(incomingEdge)
                                          .to<std::vector>();
                                for(auto const& parent : parents)
                                {
                                    if(setCoordChildren.count(parent) == 1
                                       && forLoopSequenceChildren.count(parent) == 0)
                                    {
                                        graph.control.deleteElement(incomingEdge);
                                    }
                                }
                            }
                        }
                    }

                    // Connect setCoord to the fused loop
                    if(graph.control.getOutputNodeIndices<Body>(coordPath->first)
                           .to<std::vector>()
                           .empty())
                    {
                        graph.control.deleteElement<Body>(std::vector<int>{tag},
                                                          std::vector<int>{coordPath->first});
                        graph.control.deleteElement(coordPath->first);
                        if(coordPath == paths.begin())
                            graph.control.addElement(Body(), {tag}, {fusedLoopTag});
                    }
                    else
                    {
                        graph.control.addElement(Sequence(), {coordPath->first}, {fusedLoopTag});
                    }
                }
            }
        }

        KernelGraph fuseLoops(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");
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

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
        struct FuseLoopsVisitor : public BaseGraphVisitor
        {
            FuseLoopsVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context, Graph::Direction::Upstream, false)
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            /**
             * @brief Find a path from a node to a ForLoopOp using only Sequence edges
             *
             * Returns an empty vector if no path is found.
             *
             * Only looks for a direct path, so if there are more than one Sequence
             * edges leaving a node, it will not return a path.
             *
             * @param graph
             * @param start
             * @return std::vector<int>
             */
            std::vector<int> pathToForLoop(KernelGraph& graph, int start)
            {
                std::vector<int> result;

                int currentNode = start;
                while(currentNode >= 0)
                {
                    result.push_back(currentNode);

                    if(isOperation<ForLoopOp>(graph.control.getElement(currentNode)))
                        return result;

                    auto nextNodes = graph.control.getOutputNodeIndices<Sequence>(currentNode)
                                         .to<std::vector>();

                    if(nextNodes.size() != 1)
                        currentNode = -1;
                    else
                        currentNode = nextNodes[0];
                }

                return {};
            }

            /**
             * @brief Insert Set Coordinate nodes underneath fused loop
             *
             * @tparam EdgeType
             * @param graph
             * @param forLoopTag
             * @param origSetCoord
             * @param fusedLoopTag
             */
            template <CControlEdge EdgeType>
            std::vector<int> insertSetCoordinates(KernelGraph& graph,
                                                  int          forLoopTag,
                                                  int          origSetCoord,
                                                  int          fusedLoopTag)
            {
                auto children = graph.control.getOutputNodeIndices<EdgeType>(forLoopTag)
                                    .template to<std::vector>();
                if(children.empty())
                    return {};

                auto setCoord = graph.control.addElement(graph.control.getElement(origSetCoord));
                graph.control.addElement(EdgeType(), {fusedLoopTag}, {setCoord});

                for(auto const& child : children)
                {
                    graph.control.addElement(Body(), {setCoord}, {child});
                    graph.control.deleteElement<EdgeType>(std::vector<int>{forLoopTag},
                                                          std::vector<int>{child});
                    for(auto const& c : graph.mapper.getConnections(origSetCoord))
                    {
                        graph.mapper.connect(setCoord, c.coordinate, c.connection);
                    }
                }

                return children;
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        ForLoopOp const&   op) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto newTag = reindexer.control.at(tag);
                auto bodies = graph.control.getOutputNodeIndices<Body>(newTag).to<std::set>();

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
                for(auto const& path : paths)
                {
                    if(path.second.size() != 1)
                        return;

                    auto forLoop = path.second[0].back();
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

                // Insert SetCoord between each Body and Sequence Node
                auto coordPath    = paths.begin();
                auto origSetCoord = coordPath->first;
                auto forLoopTag   = coordPath->second[0].back();

                // The first ForLoop will be the one that all of the others are fused into
                auto fusedLoopTag = forLoopTag;

                // Iterate through the rest of the for loops to be fused
                for(; coordPath != paths.end(); ++coordPath)
                {
                    auto origSetCoord = coordPath->first;
                    auto forLoopTag   = coordPath->second[0].back();

                    // For each node (N) leaving for loop with Body or Sequence edge:
                    // Delete edge from original for loop to N
                    // Add edge from new for loop to SetCoord. Add edge from SetCoord to N
                    auto sequenceNodes = insertSetCoordinates<Sequence>(
                        graph, forLoopTag, origSetCoord, fusedLoopTag);
                    insertSetCoordinates<Body>(graph, forLoopTag, origSetCoord, fusedLoopTag);

                    // For each parent (N) of for loop
                    // Delete edge from N to original for loop
                    auto pathSize = coordPath->second[0].size();
                    if(pathSize > 1)
                    {
                        auto N = coordPath->second[0][pathSize - 2];
                        graph.control.deleteElement<Sequence>(std::vector<int>{N},
                                                              std::vector<int>{forLoopTag});
                    }
                    else
                    {
                        graph.control.deleteElement<Body>(std::vector<int>{coordPath->first},
                                                          std::vector<int>{forLoopTag});
                    }

                    // Delete all Sequence edges from children of SetCoord to children of fusedLoopTag
                    auto forLoopSequenceChildren
                        = graph.control.depthFirstVisit(sequenceNodes).to<std::set>();
                    auto setCoordChildren
                        = graph.control.depthFirstVisit(coordPath->first).to<std::set>();

                    for(auto const& setCoordChild : setCoordChildren)
                    {
                        if(forLoopSequenceChildren.count(setCoordChild) == 1)
                        {
                            auto location = graph.control.getLocation(setCoordChild);
                            for(auto const& incomingEdge : location.incoming)
                            {
                                int parent
                                    = *graph.control
                                           .getNeighbours<Graph::Direction::Upstream>(incomingEdge)
                                           .begin();
                                if(setCoordChildren.count(parent) == 1
                                   && forLoopSequenceChildren.count(parent) == 0)
                                {
                                    graph.control.deleteElement(incomingEdge);
                                }
                            }
                        }
                    }

                    // Connect setCoord to the fused loop
                    if(graph.control.getOutputNodeIndices<Body>(coordPath->first)
                           .to<std::vector>()
                           .empty())
                    {
                        graph.control.deleteElement<Body>(std::vector<int>{newTag},
                                                          std::vector<int>{coordPath->first});
                        graph.control.deleteElement(coordPath->first);
                        if(coordPath == paths.begin())
                            graph.control.addElement(Body(), {newTag}, {fusedLoopTag});
                    }
                    else
                    {
                        graph.control.addElement(Sequence(), {coordPath->first}, {fusedLoopTag});
                    }

                    if(coordPath != paths.begin())
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
            }
        };

        KernelGraph fuseLoops(KernelGraph const& k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::fuseLoops");
            auto visitor = FuseLoopsVisitor(context);
            return rewrite(k, visitor);
        }
    }
}

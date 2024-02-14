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
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;
            using GD     = rocRoller::Graph::Direction;
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

            using LoadKey = std::vector<unsigned int>;

            /**
             * @brief Return information about a load.
             *
             * Returns a key that describes the data being loaded by the provided load
             * instruction. This key includes the User variable plus any information
             * from SetCoordinate nodes above the load.
             *
             * Also returns the node tag of the top SetCoordinate node above the load.
             * If there is no SetCoordinate node, returns the tag of the load.
             *
             * @param graph
             * @param load
             * @return std::pair<LoadKey, int>
             */
            std::pair<LoadKey, int> getLoadPair(KernelGraph const& graph, int load)
            {
                LoadKey key;
                int     tag = load;

                AssertFatal(graph.control.get<LoadTiled>(load), "Expecting a LoadTiled node");
                AssertFatal(graph.mapper.get<User>(load), "Load tiled missing User dimension");

                key.push_back(graph.mapper.get<User>(load));

                // Search for nested SetCoordinates
                while(true)
                {
                    auto parent = only(graph.control.getInputNodeIndices<Body>(tag));
                    if(!parent)
                        break;

                    if(graph.control.get<SetCoordinate>(*parent))
                        tag = *parent;
                    else
                        break;

                    auto setCoord = graph.control.get<SetCoordinate>(tag);
                    AssertFatal(graph.mapper.get<Unroll>(tag),
                                "SetCoordinate needs Unroll dimension");

                    key.push_back(graph.mapper.get<Unroll>(tag));
                    AssertFatal(
                        evaluationTimes(setCoord->value)[Expression::EvaluationTime::Translate],
                        "Unroll value should be a literal");
                    key.push_back(getUnsignedInt(evaluate(setCoord->value)));
                }

                return {key, tag};
            }

            /**
             * @brief Remove duplicate loads from the graph.
             *
             * Check to see if any load within loads has already been seen. If it has, it can be replaced by the
             * previously seen load.
             *
             * @param graph
             * @param seenLoads Loads that have already been seen
             * @param loads
             */
            void removeDuplicateLoads(KernelGraph&                            graph,
                                      std::map<LoadKey, std::pair<int, int>>& seenLoads,
                                      std::vector<int> const&                 loads)
            {
                for(auto const& load : loads)
                {
                    auto loadKey = getLoadPair(graph, load);

                    if(seenLoads.count(loadKey.first) > 0)
                    {
                        auto existingLoad  = seenLoads[loadKey.first];
                        auto loadMacroTile = graph.mapper.get<MacroTile>(load);
                        auto existingLoadMacroTile
                            = graph.mapper.get<MacroTile>(seenLoads[loadKey.first].first);

                        // Check to see if the MacroTile associated with the load is the "Original" MacroTile.
                        // If it is, it shouldn't be replaced. Instead, replace the already seen load.
                        auto duplicateStorage = only(graph.coordinates.getOutputNodeIndices(
                            loadMacroTile, CT::isEdge<PassThrough>));
                        if(!duplicateStorage)
                        {
                            auto tmp                 = loadKey.second;
                            loadMacroTile            = existingLoadMacroTile;
                            loadKey.second           = seenLoads[loadKey.first].second;
                            seenLoads[loadKey.first] = {load, tmp};
                            existingLoadMacroTile    = graph.mapper.get<MacroTile>(load);
                        }

                        for(auto const& c : graph.mapper.getCoordinateConnections(loadMacroTile))
                        {
                            graph.mapper.disconnect(c.control, c.coordinate, c.connection);
                            graph.mapper.connect(c.control, existingLoadMacroTile, c.connection);
                        }
                        replaceWith(graph, loadKey.second, seenLoads[loadKey.first].second, false);
                        purgeNodeAndChildren(graph, loadKey.second);

                        // MacroTile should always be a duplicate pointing to an "Original" MacroTile.
                        // Delete the edges from the MacroTile to the original, then delete the MacroTile.
                        if(graph.mapper.getCoordinateConnections(loadMacroTile).empty())
                        {
                            for(auto const& child :
                                graph.coordinates.getLocation(loadMacroTile).outgoing)
                            {
                                graph.coordinates.deleteElement(child);
                            }
                            graph.coordinates.deleteElement(loadMacroTile);
                        }
                    }
                    else
                    {
                        seenLoads[loadKey.first] = {load, loadKey.second};
                    }
                }
            }

            void fuseLoops(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops({})", tag);
                auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();

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
                    auto forLoopDim = getSize(std::get<Dimension>(graph.coordinates.getElement(
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

                std::map<LoadKey, std::pair<int, int>> seenLoads;

                for(auto const& forLoopTag : forLoopsToFuse)
                {
                    auto loads = filter(graph.control.isElemType<LoadTiled>(),
                                        graph.control.depthFirstVisit(
                                            graph.control.getOutputNodeIndices<Body>(forLoopTag)
                                                .to<std::vector>(),
                                            dontWalkPastForLoop,
                                            GD::Downstream))
                                     .to<std::vector>();

                    if(forLoopTag == fusedLoopTag)
                    {
                        removeDuplicateLoads(graph, seenLoads, loads);
                        continue;
                    }

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

                    removeDuplicateLoads(graph, seenLoads, loads);

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
                    = graph.control.getOutputNodeIndices<Sequence>(fusedLoopTag).to<std::vector>();

                auto loads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(
                                        children, dontWalkPastForLoop, GD::Downstream))
                                 .to<std::vector>();

                auto stores = filter(graph.control.isElemType<StoreTiled>(),
                                     graph.control.depthFirstVisit(
                                         children, dontWalkPastForLoop, GD::Downstream))
                                  .to<std::vector>();

                orderMemoryNodes(graph, loads, true);
                orderMemoryNodes(graph, stores, true);
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

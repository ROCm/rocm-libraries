#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Sequentialize.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Sequential Epilogue Transformation
         *
         * This transformation changes:
         *
         *     ForLoopK
         * |            |               |
         * Sequence     Sequence  ...   Sequence
         * |            |               |
         * Epilogue     Epilogue        Epilogue
         *
         * To
         * 
         * ForLoopK
         * |
	 * Sequence
	 * |
         * Scope -> Sequence ->  Scope -> ... Sequence -> Scope
         * |                     |                        |
         * Epilogue              Epilogue                 Epilogue
         */
        namespace SequentializeNS
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;
            using GD     = rocRoller::Graph::Direction;
            /**
             * @brief Generates a vector of epilogue indicators in the order based on the unroll value of their respective setCoordinates
             *
             * @param graph
             * @param std::vector<int> stores
	     * @return std::vector<int> indicators
             */
            std::vector<int> getOrderedIndicators(KernelGraph&            graph,
                                                  const std::vector<int>& stores)
            {
                auto dontWalkPastForLoop = [&](int tag) -> bool {
                    for(auto neighbour : graph.control.getNeighbours(tag, GD::Upstream))
                    {
                        if(graph.control.get<ForLoopOp>(neighbour))
                        {
                            return false;
                        }
                    }
                    return true;
                };

                auto dontWalkPastAssign = [&](int tag) -> bool {
                    for(auto neighbour : graph.control.getNeighbours(tag, GD::Upstream))
                    {
                        if(graph.control.get<Assign>(neighbour))
                        {
                            return false;
                        }
                    }
                    return true;
                };

                //TODO:
                //This predicate is GEMM specific!
                auto isIndicator = [&](int tag) -> bool {
                    auto elem = graph.control.get<Assign>(tag);
                    if(elem)
                    {
                        return (
                            std::holds_alternative<Expression::MultiplyAdd>(*(elem->expression)));
                    }
                    return false;
                };

                std::map<std::vector<unsigned int>, int> epilogueCoords;
                for(const auto& store : stores)
                {
                    auto setCoords
                        = filter(
                              graph.control.isElemType<SetCoordinate>(),
                              graph.control.depthFirstVisit(
                                  graph.control.getInputNodeIndices<Body>(store).to<std::vector>(),
                                  dontWalkPastAssign,
                                  GD::Upstream))
                              .to<std::vector>();

                    auto indicators
                        = filter(
                              isIndicator,
                              graph.control.depthFirstVisit(
                                  graph.control.getInputNodeIndices<Body>(store).to<std::vector>(),
                                  dontWalkPastForLoop,
                                  GD::Upstream))
                              .to<std::vector>();
                    AssertFatal(indicators.size() == 1,
                                "Should have only one indicator per epilogue block.");
                    std::vector<unsigned int> unrollCoords;
                    for(auto const& setCoord : setCoords)
                    {
                        auto node = graph.control.get<SetCoordinate>(setCoord);
                        unrollCoords.push_back(getUnsignedInt(evaluate(node->value)));
                    }
                    epilogueCoords[unrollCoords] = indicators[0];
                }

                std::vector<int> indicators;
                for(auto const& epilogue : epilogueCoords)
                    indicators.push_back(epilogue.second);
                return indicators;
            }

            /**
             * @brief Find all paths from a node to a Epilopue Indicators using only Sequence edges
             *
             * Returns an empty map if no paths are  found.
             *
             * @param graph
             * @param start
             * @return std::map<int, std::vector<int>>
             */
            std::map<int, std::vector<int>> pathToEpilogueIndicator(KernelGraph&      graph,
                                                                    std::vector<int>& allIndicators,
                                                                    int               start)
            {
                // Find the paths to the MultiplyAdds
                std::map<int, std::vector<int>> pathsToIndicators;
                for(auto indicator : allIndicators)
                {
                    // Find all of the nodes in between the node and the first for loop
                    auto pathToAssign = graph.control
                                            .path<GD::Downstream>(std::vector<int>{start},
                                                                  std::vector<int>{indicator})
                                            .to<std::vector>();
                    pathsToIndicators[indicator] = pathToAssign;
                }
                return pathsToIndicators;
            }

            void sequentialize(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::Sequentialize({})", tag);
                //Right now we only do this for the KLOOP, this is Specific to GEMM
                //TODO: Generalize beyond GEMM.
                auto loopNode = graph.control.get<ForLoopOp>(tag);
                if(loopNode->loopName != rocRoller::KLOOP)
                {
                    return;
                }

                //This portion of the code finds all the loads and stores of the Epilogues
                //and eliminates Sequence edges between connected via SetCoordinates.
                auto forChildren
                    = graph.control.getOutputNodeIndices<Sequence>(tag).to<std::vector>();

                auto stores = filter(graph.control.isElemType<StoreTiled>(),
                                     graph.control.depthFirstVisit(forChildren, GD::Downstream))
                                  .to<std::vector>();
                //Indicates only one epilogue.
                //This is specific to GEMMs. Here we have one store per epilogue..
                //TODO: Generalize beyond GEMM.

                if(stores.size() <= 1)
                    return;

                auto loads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(forChildren, GD::Downstream))
                                 .to<std::vector>();
                std::vector<int> setCoords;

                for(const auto& store : stores)
                {
                    setCoords.push_back(getTopSetCoordinate(graph, store));
                }
                for(const auto& load : loads)
                {
                    setCoords.push_back(getTopSetCoordinate(graph, load));
                }

                for(const auto& setCoord : setCoords)
                {
                    auto setCoordChildren
                        = graph.control.getOutputNodeIndices<Sequence>(setCoord).to<std::vector>();
                    for(const auto& child : setCoordChildren)
                    {
                        if(isOperation<SetCoordinate>(graph.control.getElement(child)))
                        {
                            graph.control.deleteElement<Sequence>(std::vector<int>{setCoord},
                                                                  std::vector<int>{child});
                        }
                    }
                }

                //Fetch the epilogue indicators in order.
                auto epilogueIndicators = getOrderedIndicators(graph, stores);
                //Fetch the paths to the nodes that indicate an Epilogue, these will be separated since
                //we eliminated the connections between the Epilogue blocks' memory nodes
                auto epiloguePaths = pathToEpilogueIndicator(graph, epilogueIndicators, tag);

                //Check epiloguePaths against Nodes from forK to select common Paths
                std::map<int, std::set<int>> nodesToSequence;

                for(auto const& epiloguePath : epiloguePaths)
                {
                    auto          epilogueVec = epiloguePath.second;
                    std::set<int> commonNodes;
                    for(auto const& child : forChildren)
                    {
                        if(std::find(epilogueVec.begin(), epilogueVec.end(), child)
                           != epilogueVec.end())
                        {
                            commonNodes.insert(child);
                        }
                    }
                    nodesToSequence[epiloguePath.first] = commonNodes;
                }

                //Delete the original Epilogue Sequence Edges from the forK loop.
                for(auto const& child : forChildren)
                {
                    graph.control.deleteElement<Sequence>(std::vector<int>{tag},
                                                          std::vector<int>{child});
                }

                //Connect Epilogue Blocks via Scopes and Sequences to
                //"Sequentialize" the Epilogues
                auto firstScope = graph.control.addElement(Scope());
                graph.control.addElement(Sequence(), {tag}, {firstScope});
                auto prevScope = firstScope;

                for(auto indicator = epilogueIndicators.begin();
                    indicator < epilogueIndicators.end();
                    indicator++)
                {
                    for(auto const& node : nodesToSequence[*indicator])
                    {
                        graph.control.addElement(Body(), {prevScope}, {node});
                    }
                    if(std::next(indicator) != epilogueIndicators.end())
                    {
                        auto nextScope = graph.control.addElement(Scope());
                        graph.control.addElement(Sequence(), {prevScope}, {nextScope});
                        prevScope = nextScope;
                    }
                }
            }
        }

        KernelGraph Sequentialize::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::Sequentialize");

            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    SequentializeNS::sequentialize(newGraph, node);
                }
            }

            return newGraph;
        }
    }
}

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderEpilogueBlocks.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Order Epilogue Blocks Transformation
         *
         * This transformation changes:
         *
         *     ForLoopY
         * |            |               |
         * Body        Body     ...    Body
         * |            |               |
         * Epilogue     Epilogue        Epilogue
         *
         * To
         *
         * ForLoopY
         * |
         * Body
         * |
         * Scope -> Sequence ->  Scope -> ... Sequence -> Scope
         * |                     |                        |
         * Body                 Body                     Body
         * |                     |                        |
         * Epilogue            Epilogue                 Epilogue
         */
        namespace OrderEpilogueBlocksNS
        {
            using GD = rocRoller::Graph::Direction;

            void orderEpilogueBlocks(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::OrderEpilogueBlocks({})", tag);
                //Right now we only do this for the YLOOP, this is Specific to GEMM
                //TODO: Generalize beyond GEMM.
                auto loopNode = graph.control.get<ForLoopOp>(tag);
                if(loopNode->loopName != rocRoller::YLOOP)
                {
                    return;
                }

                auto forChildren = graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();

                auto stores = filter(graph.control.isElemType<StoreTiled>(),
                                     graph.control.depthFirstVisit(forChildren, GD::Downstream))
                                  .to<std::vector>();
                //Indicates only one epilogue or not an epilogue at all.
                //This is specific to GEMMs. Here we have one store per epilogue.
                //TODO: Generalize beyond GEMM.

                if(stores.size() <= 1 || forChildren.size() <= 1)
                    return;

                //This portion of the code finds all the loads and stores of the Epilogues
                //and eliminates Sequence edges between connected via SetCoordinates.
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

                //Delete the original Epilogue Body Edges from the loop.
                for(auto const& child : forChildren)
                {
                    graph.control.deleteElement<Body>(std::vector<int>{tag},
                                                      std::vector<int>{child});
                }

                //Connect Epilogue Blocks via Scopes and Sequences to
                //order the Epilogues
                auto firstScope = graph.control.addElement(Scope());
                graph.control.addElement(Body(), {tag}, {firstScope});
                auto prevScope = firstScope;

                for(auto i = forChildren.begin(); i != forChildren.end(); i++)
                {
                    graph.control.addElement(Body(), {prevScope}, {*i});

                    if(std::next(i) != forChildren.end())
                    {
                        auto nextScope = graph.control.addElement(Scope());
                        graph.control.addElement(Sequence(), {prevScope}, {nextScope});
                        prevScope = nextScope;
                    }
                }
            }
        }

        KernelGraph OrderEpilogueBlocks::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::OrderEpilogueBlocks");
            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    OrderEpilogueBlocksNS::orderEpilogueBlocks(newGraph, node);
                }
            }

            return newGraph;
        }
    }
}

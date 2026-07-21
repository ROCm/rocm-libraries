// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            using namespace ControlGraph;

            /**
             * @brief Delete Body edges from a parent node to all its children
             *
             *
             *
             * @param graph The kernel graph
             * @param parent The parent node
             * @param children The children nodes
             */
            void deleteBodyEdges(KernelGraph& graph, int parent, std::vector<int> const& children)
            {
                for(auto const& child : children)
                {
                    graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                      std::vector<int>{child});
                    Log::debug("  Deleted Body edge from {} to {}", parent, child);
                }
            }

            /**
             * @brief Create a chain of Scopes connected by Sequence edges
             *
             *
             * Creates: parent -> Body -> Scope -> Sequence -> Scope -> Sequence -> ... -> Scope
             *                             |                   |                          |
             *                           Body                Body                        Body
             *                             |                   |                          |
             *                          children[0]         children[1]               children[n]
             *
             * @param graph The kernel graph
             * @param parent The parent node to attach the first Scope to
             * @param children The children nodes to place under each Scope
             */
            void createScopeChain(KernelGraph& graph, int parent, std::vector<int> const& children)
            {
                if(children.empty())
                    return;

                auto firstScope = graph.control.addElement(Scope());
                graph.control.addElement(Body(), {parent}, {firstScope});
                auto prevScope = firstScope;

                Log::debug(
                    "  Creating scope chain: parent {} -> {} scopes", parent, children.size());

                for(auto i = children.begin(); i != children.end(); i++)
                {
                    graph.control.addElement(Body(), {prevScope}, {*i});

                    if(std::next(i) != children.end())
                    {
                        auto nextScope = graph.control.addElement(Scope());
                        graph.control.addElement(Sequence(), {prevScope}, {nextScope});
                        prevScope = nextScope;
                    }
                }
            }

            /**
             * @brief Check if a ForLoopOp is or is contained within a RECEIVE loop
             */
            bool isWithinReceiveLoop(KernelGraph const& graph, int forLoopTag)
            {
                auto containingForLoops = graph.control.nodesContaining(forLoopTag)
                                              .filter(graph.control.isElemType<ForLoopOp>());

                for(auto containingLoop : containingForLoops)
                {
                    if(getForLoopName(const_cast<KernelGraph&>(graph), containingLoop)
                       == rocRoller::RECEIVE)
                    {
                        return true;
                    }
                }
                return false;
            }

            /**
             * @brief Order fixup chains within unrolled RECEIVE loops
             *
             * This transformation finds fixup chains across different unrolled iterations
             * and orders them sequentially using Scopes and Sequence edges, similar to
             * how orderEpilogueBlocks orders epilogue blocks. This change reduces the register pressure
             * by avoiding overlapping the fixup chains.
             *
             * Before:
             *   ForLoop (within RECEIVE)
             *   |           |           |
             *   Body       Body       Body
             *   |           |           |
             *   Fixup1     Fixup2     Fixup3
             *
             * After:
             *   ForLoop (within RECEIVE)
             *   |
             *   Body
             *   |
             *   Scope --Sequence--> Scope --Sequence--> Scope
             *   |                   |                   |
             *   Body               Body               Body
             *   |                   |                   |
             *   Fixup1            Fixup2            Fixup3
             */
            void orderFixupChains(KernelGraph& graph, int tag)
            {
                Log::debug("OrderEpilogueBlocks::orderFixupChains({})", tag);

                // Only order fixup chains for loops within RECEIVE
                if(getForLoopName(graph, tag) != rocRoller::YLOOP
                   || !isWithinReceiveLoop(graph, tag))
                {
                    return;
                }

                // Get all Body children of this ForLoop (these are the unrolled iterations)
                auto forChildren = graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();

                if(forChildren.size() <= 1)
                {
                    Log::debug("  Only {} body children, no need to order", forChildren.size());
                    return;
                }

                // Sort children by their execution order in the control graph
                std::sort(forChildren.begin(), forChildren.end(), [&](int a, int b) {
                    auto order = graph.control.compareNodes(UpdateCache, a, b);
                    return order == NodeOrdering::LeftFirst;
                });

                for(auto child : forChildren)
                {
                    Log::debug("  Sorted child: {}", child);
                }

                // Delete the sequence edges between consecutive SetCoordinate operations
                for(size_t i = 0; i + 1 < forChildren.size(); i++)
                {
                    auto currentNode = forChildren[i];
                    auto nextNode    = forChildren[i + 1];

                    // Verify nodes are SetCoordinate operations
                    auto currentSetCoordinate = graph.control.get<SetCoordinate>(currentNode);
                    auto nextSetCoordinate    = graph.control.get<SetCoordinate>(nextNode);
                    AssertFatal(currentSetCoordinate && nextSetCoordinate,
                                "SetCoordinate operation not found");

                    // Verify there is a sequence edge between nodes before deletion
                    auto neighbors = graph.control.getOutputNodeIndices<Sequence>(currentNode)
                                         .to<std::vector>();
                    if(std::find(neighbors.begin(), neighbors.end(), nextNode) == neighbors.end())
                        Throw<FatalError>("Sequence edge not found");

                    graph.control.deleteElement<Sequence>(std::vector<int>{currentNode},
                                                          std::vector<int>{nextNode});
                    Log::debug("  Deleted Sequence edge from {} to {}", currentNode, nextNode);
                }

                // Delete the original Body edges from the ForLoop to the bodies
                deleteBodyEdges(graph, tag, forChildren);

                // Create a chain of Scopes connected by Sequence edges
                createScopeChain(graph, tag, forChildren);
            }

            void orderEpilogueBlocks(KernelGraph& graph, int tag, UnrollColouring const& colouring)
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

                for(auto t : forChildren)
                {
                    Log::debug("  Ordering under {}", t);
                }

                auto reachable = graph.control.depthFirstVisit(forChildren, GD::Downstream)
                                     .to<std::unordered_set>();

                for(auto k : colouring.separators)
                {
                    if(reachable.contains(k))
                    {
                        Log::debug("  Deleting separator {}", k);
                        graph.control.deleteElement(k);
                    }
                }

                //Delete the original Epilogue Body Edges from the loop.
                deleteBodyEdges(graph, tag, forChildren);

                //Connect Epilogue Blocks via Scopes and Sequences to
                //order the Epilogues
                createScopeChain(graph, tag, forChildren);
            }
        }

        KernelGraph OrderEpilogueBlocks::apply(KernelGraph const& original)
        {
            auto graph = original;

            // Exclude Unrolls that aren't associated with
            // JammedWaveTileNumbers from the colouring.
            auto exclude = std::unordered_set<int>();
            for(auto unroll : graph.coordinates.getNodes<CoordinateGraph::Unroll>())
            {
                bool isJammed = false;
                for(auto input :
                    graph.coordinates.getInputNodeIndices(unroll, [](auto) { return true; }))
                {
                    auto maybeJammed
                        = graph.coordinates.get<CoordinateGraph::JammedWaveTileNumber>(input);
                    if(maybeJammed)
                        isJammed = true;
                }

                if(!isJammed)
                    exclude.insert(unroll);
            }
            auto colouring = colourByUnrollValue(original, -1, exclude);

            for(const auto node : graph.control.getNodes<ControlGraph::ForLoopOp>())
            {
                OrderEpilogueBlocksNS::orderEpilogueBlocks(graph, node, colouring);
                OrderEpilogueBlocksNS::orderFixupChains(graph, node);
            }

            return graph;
        }
    }
}

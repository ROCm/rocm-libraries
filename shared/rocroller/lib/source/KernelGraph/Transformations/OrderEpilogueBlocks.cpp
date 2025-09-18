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
            }

            return graph;
        }
    }
}

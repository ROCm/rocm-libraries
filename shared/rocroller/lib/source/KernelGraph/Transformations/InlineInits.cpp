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

#include <iterator>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineInits.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using GD = rocRoller::Graph::Direction;

        KernelGraph InlineInits::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::inlineInits");
            auto graph = original;

            auto sequenceLeaves
                = graph.control.getNodes()
                      .filter([&](int nodeIdx) {
                          return graph.control.getOutputNodeIndices<Sequence>(nodeIdx).empty();
                      })
                      .to<std::set>();

            for(auto nodeIdx : graph.control.getNodes())
            {
                auto node = graph.control.getNode(nodeIdx);
                if(std::holds_alternative<ForLoopOp>(node)
                   || std::holds_alternative<ConditionalOp>(node))
                    continue;

                std::unordered_set<int> leaves;

                auto connectedByInit
                    = graph.control.getOutputNodeIndices<Initialize>(nodeIdx).to<std::set>();

                auto downstreamFromInit = graph.control.followEdges<Sequence>(connectedByInit);
                std::set_intersection(sequenceLeaves.begin(),
                                      sequenceLeaves.end(),
                                      downstreamFromInit.begin(),
                                      downstreamFromInit.end(),
                                      std::inserter(leaves, leaves.begin()));

                auto connectedByBody
                    = graph.control.getOutputNodeIndices<Body>(nodeIdx).to<std::unordered_set>();

                for(auto byInit : connectedByInit)
                {
                    auto initIdx = graph.control.findEdge(nodeIdx, byInit);
                    AssertFatal(initIdx);
                    graph.control.setElement(*initIdx, Body());
                }

                for(auto leaf : leaves)
                {
                    for(auto byBody : connectedByBody)
                    {
                        graph.control.addElement(Sequence(), {leaf}, {byBody});
                    }
                }
            }

            return graph;
        }

    }
}

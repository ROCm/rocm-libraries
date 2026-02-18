/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph::NodeScheduling
{
    ControlGraph::ControlGraph createSubGraph(KernelGraph const&      graph,
                                              std::vector<int> const& nodes)
    {
        ControlGraph::ControlGraph subGraph;

        for(auto node : nodes)
        {
            subGraph.setElement(node, graph.control.getElement(node));
        }

        for(auto iterA = nodes.begin(); iterA != nodes.end(); ++iterA)
        {
            for(auto iterB = iterA + 1; iterB != nodes.end(); ++iterB)
            {
                auto order = graph.control.compareNodes(UpdateCache, *iterA, *iterB);

                switch(order)
                {
                case ControlGraph::NodeOrdering::LeftFirst:
                    subGraph.addElement(ControlGraph::Sequence{}, {*iterA}, {*iterB});
                    break;
                case ControlGraph::NodeOrdering::RightFirst:
                    subGraph.addElement(ControlGraph::Sequence{}, {*iterB}, {*iterA});
                    break;
                case ControlGraph::NodeOrdering::LeftInBodyOfRight:
                    subGraph.addElement(ControlGraph::Body{}, {*iterB}, {*iterA});
                    break;
                case ControlGraph::NodeOrdering::RightInBodyOfLeft:
                    subGraph.addElement(ControlGraph::Body{}, {*iterA}, {*iterB});
                    break;
                case ControlGraph::NodeOrdering::Undefined:
                    break;
                case ControlGraph::NodeOrdering::Count:
                    Throw<FatalError>("Should not get here!");
                    break;
                }
            }
        }

        removeRedundantSequenceEdges(subGraph);

        return subGraph;
    }

}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

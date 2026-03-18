// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

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

#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveSetCoordinate.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveSetCoordinate_details.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CG = rocRoller::KernelGraph::ControlGraph;

        namespace RemoveSetCoordinateDetails
        {
            void findLeaves(int                      tag,
                            KernelGraph const&       kg,
                            std::unordered_set<int>& visited,
                            std::vector<int>&        leaves)
            {
                visited.insert(tag);

                bool hasChildren = false;

                auto traverse = [&]<typename EdgeType>() {
                    for(auto child : kg.control.getOutputNodeIndices<EdgeType>(tag))
                    {
                        hasChildren = true;
                        if(not visited.contains(child))
                        {
                            findLeaves(child, kg, visited, leaves);
                        }
                    }
                };

                traverse.template operator()<ControlGraph::Sequence>();

                if(hasChildren)
                {
                    // Consider reusing visited for direct traversal instead of calling DFS
                    for(auto child : kg.control.getOutputNodeIndices<CG::Body>(tag))
                        for(auto node : kg.control.depthFirstVisit(child))
                            visited.insert(node);
                }
                else
                    traverse.template operator()<ControlGraph::Body>();

                if(not hasChildren)
                    leaves.push_back(tag);
            }

            std::vector<int> findLeaves(std::vector<int> nodes, KernelGraph const& kg)
            {
                std::unordered_set<int> visited;
                std::vector<int>        leaves;

                for(auto node : nodes)
                {
                    findLeaves(node, kg, visited, leaves);
                }
                return leaves;
            }

            void removeSetCoordinates(KernelGraph& kg)
            {
                using GD = rocRoller::Graph::Direction;

                auto const setCoordinates = kg.control.getNodes<CG::SetCoordinate>().to<std::set>();
                for(auto const sc : setCoordinates | std::views::reverse)
                {
                    auto inputNodes = kg.control.getInputNodeIndices(sc, [](auto) { return true; })
                                          .to<std::vector>();
                    AssertFatal(not inputNodes.empty());

                    auto const checkInputEdgeType = [&]<typename EdgeType>(int node) {
                        return std::ranges::all_of(
                            kg.control.getNeighbours<GD::Upstream>(node), [&](int x) {
                                return std::holds_alternative<EdgeType>(kg.control.getEdge(x));
                            });
                    };

                    bool const                        areAllInputsSequenceEdge
                        = checkInputEdgeType.template operator()<CG::Sequence>(sc);
                    if(not areAllInputsSequenceEdge)
                    {
                        bool const                        areAllInputsBodyEdge
                            = checkInputEdgeType.template operator()<CG::Body>(sc);
                        AssertFatal(areAllInputsBodyEdge,
                                    "The input edges of a SetCoordinate are not Sequence nor Body");
                    }

                    auto bodyNodes
                        = kg.control.getOutputNodeIndices<CG::Body>(sc).to<std::vector>();
                    auto sequenceNodes
                        = kg.control.getOutputNodeIndices<CG::Sequence>(sc).to<std::vector>();

                    deleteControlNode(kg, sc);

                    if(not bodyNodes.empty())
                    {
                        if(areAllInputsSequenceEdge)
                            connectAllPairs<CG::Sequence>(inputNodes, bodyNodes, kg);
                        else
                            connectAllPairs<CG::Body>(inputNodes, bodyNodes, kg);

                        connectAllPairs<CG::Sequence>(findLeaves(bodyNodes, kg), sequenceNodes, kg);
                    }
                    else
                    {
                        if(areAllInputsSequenceEdge)
                            connectAllPairs<CG::Sequence>(inputNodes, sequenceNodes, kg);
                        else
                            connectAllPairs<CG::Body>(inputNodes, sequenceNodes, kg);
                    }
                }
            }
        }

        KernelGraph RemoveSetCoordinate::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::RemoveSetCoordinate");

            auto newGraph = k;

            newGraph.buildAllTransformers();
            RemoveSetCoordinateDetails::removeSetCoordinates(newGraph);

            // Post-transformation check: should NOT have any SetCoordinates in Control Graph
            auto setCoordinates
                = newGraph.control.getElements<CG::SetCoordinate>().to<std::vector>();
            AssertFatal(setCoordinates.empty(),
                        "Control graph still has SetCoordinates: ",
                        ShowValue(setCoordinates));

            newGraph.setRestricted();

            return newGraph;
        }
    }
}

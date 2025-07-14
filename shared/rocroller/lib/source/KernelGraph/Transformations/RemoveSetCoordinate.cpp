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

#include <rocRoller/Context.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveSetCoordinate.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CG = rocRoller::KernelGraph::ControlGraph;

        static bool
            isParentSetCoordinate(CG::ControlGraph const& graph, int const edge, int const node)
        {
            return std::holds_alternative<CG::SetCoordinate>(graph.getNode(node))
                   && (std::holds_alternative<CG::Initialize>(graph.getEdge(edge))
                       || std::holds_alternative<CG::Body>(graph.getEdge(edge)));
        }

        static bool isParentForLoopOp(CG::ControlGraph const& graph, int const edge, int const node)
        {
            return std::holds_alternative<CG::ForLoopOp>(graph.getNode(node))
                   && (std::holds_alternative<CG::Body>(graph.getEdge(edge))
                       || std::holds_alternative<CG::ForLoopIncrement>(graph.getEdge(edge)));
        }

        static void buildControlStack(int                           tag,
                                      std::unordered_map<int, int>& controlStack,
                                      CG::ControlGraph const&       graph)
        {
            using GD = rocRoller::Graph::Direction;

            int parent = -1;

            auto const edge = graph.getNeighbours<Graph::Direction::Upstream>(tag).take(1).only();
            if(edge.has_value())
            {
                auto const node
                    = graph.getNeighbours<Graph::Direction::Upstream>(*edge).take(1).only();
                AssertFatal(node.has_value(), "Node does not exist!");

                if(isParentSetCoordinate(graph, *edge, *node)
                   or isParentForLoopOp(graph, *edge, *node))
                    parent = *node;
                else
                {
                    if(not controlStack.contains(*node))
                        buildControlStack(*node, controlStack, graph);
                    parent = controlStack.at(*node);
                }
            }

            controlStack[tag] = parent;
        }

        static std::unordered_map<int, int> buildControlStack(KernelGraph const& kg)
        {
            std::unordered_map<int, int> controlStack;

            for(auto const node : kg.control.getNodes())
            {
                if(not controlStack.contains(node))
                    buildControlStack(node, controlStack, kg.control);
            }

            return controlStack;
        }

        static void collectSetCoordinates(KernelGraph const&       graph,
                                          std::unordered_set<int>& visited,
                                          int                      tag,
                                          std::vector<int>&        result)
        {
            auto traverse = [&]<typename... EdgeTypes>()
            {
                auto traverse = [&]<typename EdgeType>() {
                    for(auto child : graph.control.getOutputNodeIndices<EdgeType>(tag))
                    {
                        if(not visited.contains(child))
                        {
                            visited.insert(child);
                            collectSetCoordinates(graph, visited, child, result);
                        }
                    }
                };
                (traverse.template operator()<EdgeTypes>(), ...);
            };

            traverse.template operator()<CG::Sequence,
                                         CG::ForLoopIncrement,
                                         CG::Else,
                                         CG::Body,
                                         CG::Initialize>();

            if(graph.control.get<CG::SetCoordinate>(tag).has_value())
                result.push_back(tag);
        }

        static std::vector<int> collectSetCoordinates(KernelGraph const& graph)
        {
            auto roots = graph.control.roots().to<std::vector>();

            std::vector<int> result;
            if(roots.empty())
                return result;

            std::unordered_set<int> visited;
            for(auto const& node : roots)
            {
                visited.insert(node);
                collectSetCoordinates(graph, visited, node, result);
            }
            return result;
        }

        static void buildTransformers(KernelGraph& kg)
        {
            auto cs = buildControlStack(kg);
            for(auto const& [node, parent] : cs)
            {
                auto [iter, _] = kg.transformers.emplace(node, &kg.coordinates);

                auto tag = parent;
                while(tag != -1)
                {
                    if(std::holds_alternative<CG::SetCoordinate>(kg.control.getNode(tag)))
                    {
                        auto connections = kg.mapper.getConnections(tag);
                        if(not iter->second.hasCoordinate(connections[0].coordinate))
                        {
                            auto setCoordinate
                                = std::get<CG::SetCoordinate>((kg.control.getNode(tag)));

                            iter->second.setCoordinate(connections[0].coordinate,
                                                       setCoordinate.value);
                        }
                    }
                    else
                    {
                        AssertFatal(
                            kg.control.isElemType<CG::ForLoopOp>()(tag),
                            "A node in control stack is not a ForLoopOp nor a SetCoordinate");

                        auto loopIncrTag = kg.mapper.get(tag, NaryArgument::DEST);
                        auto expr        = std::make_shared<Expression::Expression>(
                            rocRoller::Expression::DataFlowTag{
                                loopIncrTag, Register::Type::Scalar, rocRoller::DataType::Int32});
                        auto loopDims
                            = kg.coordinates.getOutputNodeIndices<CoordinateGraph::DataFlowEdge>(
                                loopIncrTag);
                        for(auto const& dim : loopDims)
                        {
                            if(not iter->second.hasCoordinate(dim))
                            {
                                iter->second.setCoordinate(dim, expr);
                            }
                        }
                    }

                    tag = cs.at(tag);
                }
            }
        }

        static void findLeaves(int                      tag,
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
                // This might not be necessary
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

        static std::vector<int> findLeaves(std::vector<int> nodes, KernelGraph const& kg)
        {
            std::unordered_set<int> visited;
            std::vector<int>        leaves;

            for(auto node : nodes)
            {
                findLeaves(node, kg, visited, leaves);
            }
            return leaves;
        }

        static void connect(bool const              isSequenceEdge,
                            std::vector<int> const& A,
                            std::vector<int> const& B,
                            KernelGraph&            kg)
        {
            if(A.empty() or B.empty())
                return;

            auto addEdges = [&]<typename EdgeType>() {
                for(auto const a : A)
                    for(auto const b : B)
                        kg.control.addElement(EdgeType(), {a}, {b});
            };

            if(isSequenceEdge)
                addEdges.template operator()<ControlGraph::Sequence>();
            else
                addEdges.template operator()<ControlGraph::Body>();
        }

        static bool const verifyInputEdgesAreOfTheSameType(KernelGraph const& kg, int const node)
        {
            using GD = rocRoller::Graph::Direction;

            auto const allSequence
                = std::ranges::all_of(kg.control.getNeighbours<GD::Upstream>(node), [&](int x) {
                      return std::holds_alternative<CG::Sequence>(kg.control.getEdge(x));
                  });

            if(not allSequence)
            {
                auto const allBody
                    = std::ranges::all_of(kg.control.getNeighbours<GD::Upstream>(node), [&](int x) {
                          return std::holds_alternative<CG::Body>(kg.control.getEdge(x));
                      });
                AssertFatal(allBody, "A SetCoordinate's input edges are of different types");
            }

            return allSequence;
        }

        static void removeSetCoordinates(KernelGraph& kg)
        {
            using GD = rocRoller::Graph::Direction;

            auto setCoordinates = collectSetCoordinates(kg);
            for(auto sc : setCoordinates)
            {
                auto inputNodes = kg.control.getInputNodeIndices(sc, [](auto) { return true; })
                                      .to<std::vector>();
                AssertFatal(not inputNodes.empty());

                bool const isSequenceEdge = verifyInputEdgesAreOfTheSameType(kg, sc);

                auto bodyNodes = kg.control.getOutputNodeIndices<CG::Body>(sc).to<std::vector>();
                auto sequenceNodes
                    = kg.control.getOutputNodeIndices<CG::Sequence>(sc).to<std::vector>();

                // TODO: use Matt's deleteControlNode to delete SetCoordinate
                std::vector<int> del;
                for(auto edge : kg.control.getNeighbours<GD::Upstream>(sc))
                    del.push_back(edge);
                for(auto edge : kg.control.getNeighbours<GD::Downstream>(sc))
                    del.push_back(edge);

                for(auto edge : del)
                    kg.control.deleteElement(edge);

                kg.control.deleteElement(sc);
                kg.mapper.purge(sc);

                if(not bodyNodes.empty())
                {
                    connect(isSequenceEdge, inputNodes, bodyNodes, kg);
                    connect(true, findLeaves(bodyNodes, kg), sequenceNodes, kg);
                }
                else
                    connect(isSequenceEdge, inputNodes, sequenceNodes, kg);
            }
        }

        KernelGraph RemoveSetCoordinate::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::RemoveSetCoordinate");

            auto newGraph = k;

            buildTransformers(newGraph);
            removeSetCoordinates(newGraph);

            // Post-transformation check: should NOT have any SetCoordinates in Control Graph
            auto setCoordinates
                = newGraph.control.getElements<CG::SetCoordinate>().to<std::vector>();
            AssertFatal(setCoordinates.empty(),
                        "Control graph still has SetCoordinates: ",
                        ShowValue(setCoordinates));

            return newGraph;
        }
    }
}

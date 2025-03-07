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

#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        template <std::predicate<CoordinateGraph::Dimension const&> NodePredicate,
                  std::predicate<CoordinateGraph::Edge const&>      EdgePredicate>
        std::set<int> getLeafNodesWithPredicates(CoordinateGraph::CoordinateGraph const& graph,
                                                 int                                     start,
                                                 NodePredicate nodePredicate,
                                                 EdgePredicate edgePredicate)
        {
            std::set<int> next;

            for(int node : graph.getOutputNodeIndices(start, edgePredicate))
            {
                if(nodePredicate(graph.getNode(node)))
                    next.insert(node);
            }

            if(next.empty())
            {
                return {start};
            }

            std::set<int> rv;

            for(int node : next)
            {
                auto nodeLeaves
                    = getLeafNodesWithPredicates(graph, node, nodePredicate, edgePredicate);
                rv.insert(nodeLeaves.begin(), nodeLeaves.end());
            }

            return rv;
        }

        std::set<int> loadNodesReachableWithoutDimensionModifyingNodes(
            ControlGraph::ControlGraph const& graph, int start)
        {
            auto isLoadTiled = [](ControlGraph::Operation const& op) {
                return std::holds_alternative<ControlGraph::LoadTiled>(op);
            };

            auto isSequenceEdge = [](ControlGraph::ControlEdge const& edge) {
                return std::holds_alternative<ControlGraph::Sequence>(edge);
            };

            auto isNotDimensionModifyingNode = [](ControlGraph::Operation const& op) {
                return !std::holds_alternative<ControlGraph::TensorContraction>(op);
            };

            auto sameDimensionLoadTiledNodes
                = reachableNodes<Graph::Direction::Upstream>(
                      graph, start, isNotDimensionModifyingNode, isSequenceEdge, isLoadTiled)
                      .to<std::set>();
            return sameDimensionLoadTiledNodes;
        }

        struct RedundantCommandArgsVisitor
        {

            template <typename Op>
            requires CIsAnyOf<Op,
                              ControlGraph::AssertOp,
                              ControlGraph::Assign,
                              ControlGraph::Barrier,
                              ControlGraph::Block,
                              ControlGraph::ComputeIndex,
                              ControlGraph::ConditionalOp,
                              ControlGraph::Deallocate,
                              ControlGraph::DoWhileOp,
                              ControlGraph::Exchange,
                              ControlGraph::ForLoopOp,
                              ControlGraph::Kernel,
                              ControlGraph::LoadLDSTile,
                              ControlGraph::LoadLinear,
                              ControlGraph::LoadSGPR,
                              ControlGraph::LoadTiled,
                              ControlGraph::LoadVGPR,
                              ControlGraph::LoadTileDirect2LDS,
                              ControlGraph::Multiply,
                              ControlGraph::NOP,
                              ControlGraph::Scope,
                              ControlGraph::SeedPRNG,
                              ControlGraph::SetCoordinate,
                              ControlGraph::StoreLDSTile,
                              ControlGraph::StoreLinear,
                              ControlGraph::StoreSGPR,
                              //   ControlGraph::StoreTiled,
                              ControlGraph::StoreVGPR,
                              //   ControlGraph::TensorContraction,
                              ControlGraph::UnrollOp,
                              ControlGraph::WaitZero>
            void operator()(int nodeID, Op const& op) {}

            void operator()(int nodeID, ControlGraph::StoreTiled const& op)
            {
                auto storeTile = graph.mapper.get<CoordinateGraph::MacroTile>(nodeID);
                auto isDestructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::DestructMacroTile>;
                auto storeDims
                    = graph.coordinates.getOutputNodeIndices(storeTile, isDestructMacroTile)
                          .to<std::vector>();

                auto isConstructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::ConstructMacroTile>;

                auto sameDimensionLoadTiledNodes
                    = loadNodesReachableWithoutDimensionModifyingNodes(graph.control, nodeID);

                for(int loadID : sameDimensionLoadTiledNodes)
                {
                    auto loadTile = graph.mapper.get<CoordinateGraph::MacroTile>(loadID);
                    auto loadDims
                        = graph.coordinates.getInputNodeIndices(loadTile, isConstructMacroTile)
                              .to<std::vector>();

                    AssertFatal(loadDims.size() == storeDims.size(),
                                ShowValue(loadDims.size()),
                                ShowValue(storeDims.size()));

                    for(size_t i = 0; i < loadDims.size(); i++)
                        redundantArgs.push_back({loadDims.at(i), storeDims.at(i)});
                }
            }

            void operator()(int nodeID, ControlGraph::TensorContraction const& op)
            {
                auto D = graph.mapper.get(nodeID, NaryArgument::DEST);
                AssertFatal(D > 0, ShowValue(D));

                auto A = graph.mapper.get(nodeID, NaryArgument::LHS);
                auto B = graph.mapper.get(nodeID, NaryArgument::RHS);
                AssertFatal(A > 0, ShowValue(A));
                AssertFatal(B > 0, ShowValue(B));

                auto isConstructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::ConstructMacroTile>;

                auto aTileDims = graph.coordinates.getInputNodeIndices(A, isConstructMacroTile)
                                     .to<std::vector>();
                auto bTileDims = graph.coordinates.getInputNodeIndices(B, isConstructMacroTile)
                                     .to<std::vector>();

                AssertFatal(aTileDims.size() == bTileDims.size());

                AssertFatal(op.aDims.size() == op.bDims.size(),
                            ShowValue(op.aDims.size()),
                            ShowValue(op.bDims.size()));

                std::set<int> remainingADims(aTileDims.begin(), aTileDims.end());
                std::set<int> remainingBDims(bTileDims.begin(), bTileDims.end());

                for(size_t i = 0; i < op.aDims.size(); i++)
                {
                    auto aDim = aTileDims.at(op.aDims.at(i));
                    auto bDim = bTileDims.at(op.bDims.at(i));

                    redundantArgs.push_back({aDim, bDim});
                    remainingADims.erase(aDim);
                    remainingBDims.erase(bDim);
                }

                AssertFatal(remainingADims.size() == 1, ShowValue(remainingADims.size()));
                AssertFatal(remainingBDims.size() == 1, ShowValue(remainingBDims.size()));

                auto isDataFlowEdge = CoordinateGraph::isEdge<CoordinateGraph::DataFlow>;
                auto isMacroTile    = [](CoordinateGraph::Dimension const& dim) {
                    return std::holds_alternative<CoordinateGraph::MacroTile>(dim);
                };

                auto finalDMacroTiles
                    = getLeafNodesWithPredicates(graph.coordinates, D, isMacroTile, isDataFlowEdge);

                AssertFatal(finalDMacroTiles.size() == 1, ShowValue(finalDMacroTiles.size()));

                auto isDestructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::DestructMacroTile>;
                for(int dTile : finalDMacroTiles)
                {
                    auto dTileDims
                        = graph.coordinates.getOutputNodeIndices(dTile, isDestructMacroTile)
                              .to<std::vector>();

                    AssertFatal(dTileDims.size() == 2, ShowValue(dTileDims.size()));

                    redundantArgs.push_back({*remainingADims.begin(), dTileDims.at(0)});
                    redundantArgs.push_back({*remainingBDims.begin(), dTileDims.at(1)});
                }
            }

            void call(std::variant<int> nodeID, ControlGraph::Operation const& op)
            {
                std::visit(*this, nodeID, op);
            }

            KernelGraph const&         graph;
            std::vector<std::set<int>> redundantArgs;
        };

        std::vector<std::set<int>> identifyParallelDimensionSets(KernelGraph const& graph)
        {
            RedundantCommandArgsVisitor visitor{graph};

            for(auto nodeID : graph.control.getNodes())
            {
                visitor.call(nodeID, graph.control.getNode(nodeID));
            }

            return visitor.redundantArgs;
        }

        KernelGraph IdentifyParallelDimensions::apply(KernelGraph const& original)
        {
            auto copy = original;

            auto parallelDims = mergeSets(identifyParallelDimensionSets(copy));

            for(auto const& dimSet : parallelDims)
            {
                Expression::ExpressionPtr dimSize;

                for(int dim : dimSet)
                {
                    auto const& subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(subDim->size)
                    {
                        dimSize = subDim->size;
                        break;
                    }
                }

                AssertFatal(dimSize);

                for(int dim : dimSet)
                {
                    auto subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    subDim->size = dimSize;
                    copy.coordinates.setElement(dim, *subDim);
                }
            }

            return copy;
        }
    }
}

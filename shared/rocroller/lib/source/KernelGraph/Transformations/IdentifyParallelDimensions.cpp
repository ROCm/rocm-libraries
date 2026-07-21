// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

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
                              ControlGraph::LoadTiledTDMToLDS,
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

                    Log::debug("IdentifyParallelDimensions: Matching {} dimensions between "
                               "StoreTiled node {} and LoadTiled node {}",
                               loadDims.size(),
                               nodeID,
                               loadID);

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

                auto notFixedSize = [&](int tag) -> bool {
                    auto size = getSize(graph.coordinates.getNode(tag));
                    return !Expression::evaluationTimes(
                        size)[Expression::EvaluationTime::Translate];
                };

                auto aTileDims = graph.coordinates.getInputNodeIndices(A, isConstructMacroTile)
                                     .filter(notFixedSize)
                                     .to<std::vector>();
                auto bTileDims = graph.coordinates.getInputNodeIndices(B, isConstructMacroTile)
                                     .filter(notFixedSize)
                                     .to<std::vector>();

                AssertFatal(aTileDims.size() == bTileDims.size());

                AssertFatal(op.aDims.size() == op.bDims.size(),
                            ShowValue(op.aDims.size()),
                            ShowValue(op.bDims.size()));

                // Separate dimensions into free and contracted
                // For standard GEMM: aFreeDims=[M], aContractedDims=[K], bFreeDims=[N], bContractedDims=[K]
                std::set<size_t> aContractedIndices(op.aDims.begin(), op.aDims.end());
                std::set<size_t> bContractedIndices(op.bDims.begin(), op.bDims.end());

                std::vector<int> aFreeDims;
                std::vector<int> aContractedDims;
                std::vector<int> bFreeDims;
                std::vector<int> bContractedDims;

                for(size_t i = 0; i < aTileDims.size(); ++i)
                {
                    if(aContractedIndices.contains(i))
                        aContractedDims.push_back(aTileDims[i]);
                    else
                        aFreeDims.push_back(aTileDims[i]);
                }

                for(size_t i = 0; i < bTileDims.size(); ++i)
                {
                    if(bContractedIndices.contains(i))
                        bContractedDims.push_back(bTileDims[i]);
                    else
                        bFreeDims.push_back(bTileDims[i]);
                }

                AssertFatal(aContractedDims.size() == bContractedDims.size(),
                            ShowValue(aContractedDims.size()),
                            ShowValue(bContractedDims.size()));

                Log::debug("IdentifyParallelDimensions: Matching {} contracted dims between A and "
                           "B for TensorContraction node {}",
                           aContractedDims.size(),
                           nodeID);

                for(size_t i = 0; i < aContractedDims.size(); ++i)
                    redundantArgs.push_back({aContractedDims[i], bContractedDims[i]});

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

                    size_t expectedDSize = aFreeDims.size() + bFreeDims.size();
                    AssertFatal(dTileDims.size() == expectedDSize,
                                ShowValue(dTileDims.size()),
                                ShowValue(expectedDSize),
                                ShowValue(aFreeDims.size()),
                                ShowValue(bFreeDims.size()));

                    Log::debug("IdentifyParallelDimensions: Matching {} free dims of A "
                               "and {} free dims of B to output D of TensorContraction node {}",
                               aFreeDims.size(),
                               bFreeDims.size(),
                               nodeID);

                    // Match A's free dimensions to D's first dimensions
                    for(size_t i = 0; i < aFreeDims.size(); ++i)
                        redundantArgs.push_back({aFreeDims[i], dTileDims[i]});

                    // Match B's free dimensions to D's remaining dimensions
                    for(size_t i = 0; i < bFreeDims.size(); ++i)
                        redundantArgs.push_back({bFreeDims[i], dTileDims[aFreeDims.size() + i]});
                }

                // Handle block scaled tensors
                // ScaleA dimensions: [M, K/blockSize]
                // ScaleB dimensions: [K/blockSize, N]
                auto maybeScaleA = [&]() -> std::optional<int> {
                    auto val = graph.mapper.get(nodeID, NaryArgument::LHS_SCALE);
                    return (val > 0) ? std::optional<int>(val) : std::nullopt;
                }();
                auto maybeScaleB = [&]() -> std::optional<int> {
                    auto val = graph.mapper.get(nodeID, NaryArgument::RHS_SCALE);
                    return (val > 0) ? std::optional<int>(val) : std::nullopt;
                }();

                if(maybeScaleA.has_value() || maybeScaleB.has_value())
                {
                    std::vector<int> scaleADims, scaleBDims;

                    // ScaleA present
                    if(maybeScaleA.has_value())
                    {
                        scaleADims
                            = graph.coordinates
                                  .getInputNodeIndices(maybeScaleA.value(), isConstructMacroTile)
                                  .filter(notFixedSize)
                                  .to<std::vector>();

                        // Only match dimensions if scale is not SingleScale
                        if(scaleADims.size() > 1)
                        {
                            size_t expectedScaleASize = aFreeDims.size() + aContractedDims.size();
                            AssertFatal(scaleADims.size() == expectedScaleASize,
                                        ShowValue(scaleADims.size()),
                                        ShowValue(expectedScaleASize),
                                        ShowValue(aFreeDims.size()),
                                        ShowValue(aContractedDims.size()));

                            Log::debug("IdentifyParallelDimensions: Matching {} ScaleA free dims "
                                       "with A in TensorContraction node {}",
                                       aFreeDims.size(),
                                       nodeID);

                            // Match ScaleA's free dimensions with A's free dimensions
                            for(size_t i = 0; i < aFreeDims.size(); ++i)
                                redundantArgs.push_back({scaleADims[i], aFreeDims[i]});
                        }
                    }

                    // ScaleB present
                    if(maybeScaleB.has_value())
                    {
                        scaleBDims
                            = graph.coordinates
                                  .getInputNodeIndices(maybeScaleB.value(), isConstructMacroTile)
                                  .filter(notFixedSize)
                                  .to<std::vector>();

                        // Only match dimensions if scale is not SingleScale
                        if(scaleBDims.size() > 1)
                        {
                            size_t expectedScaleBSize = bContractedDims.size() + bFreeDims.size();
                            AssertFatal(scaleBDims.size() == expectedScaleBSize,
                                        ShowValue(scaleBDims.size()),
                                        ShowValue(expectedScaleBSize),
                                        ShowValue(bContractedDims.size()),
                                        ShowValue(bFreeDims.size()));

                            Log::debug("IdentifyParallelDimensions: Matching {} ScaleB free dims "
                                       "with B in TensorContraction node {}",
                                       bFreeDims.size(),
                                       nodeID);

                            for(size_t i = 0; i < bFreeDims.size(); ++i)
                            {
                                size_t scaleBIdx = bContractedDims.size() + i;
                                redundantArgs.push_back({scaleBDims[scaleBIdx], bFreeDims[i]});
                            }
                        }
                    }

                    // ScaleA and ScaleB both present
                    if(maybeScaleA.has_value() && maybeScaleB.has_value() && scaleADims.size() > 1
                       && scaleBDims.size() > 1)
                    {
                        Log::debug(
                            "IdentifyParallelDimensions: Matching {} blocked contracted dims "
                            "between "
                            "ScaleA and ScaleB in TensorContraction node {}",
                            aContractedDims.size(),
                            nodeID);

                        for(size_t i = 0; i < aContractedDims.size(); ++i)
                        {
                            size_t scaleAIdx = aFreeDims.size() + i;
                            size_t scaleBIdx = i;
                            redundantArgs.push_back({scaleADims[scaleAIdx], scaleBDims[scaleBIdx]});
                        }
                    }
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

        /**
         * Maps a canonical expression to the list of expressions that should be replaced with it
         */
        struct ReplacementMapping
        {
            Expression::ExpressionPtr              canonical;
            std::vector<Expression::ExpressionPtr> toReplace;
        };

        /**
         * Visitor to recursively replace expressions with canonical ones following a replacement mapping
         */
        struct ReplaceExprWithExprVisitor
        {
            ReplaceExprWithExprVisitor(std::vector<ReplacementMapping> const& replacements)
                : m_replacements(replacements)
            {
            }

            template <Expression::CUnary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CBinary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CTernary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CNary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression::Expression>(std::move(cpy));
            }

            Expression::ExpressionPtr operator()(Expression::ScaledMatrixMultiply const& expr) const
            {
                Expression::ScaledMatrixMultiply cpy = expr;
                cpy.matA                             = call(expr.matA);
                cpy.matB                             = call(expr.matB);
                cpy.matC                             = call(expr.matC);
                cpy.scaleA                           = call(expr.scaleA);
                cpy.scaleB                           = call(expr.scaleB);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CValue Value>
            Expression::ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr call(Expression::ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                // Find if this expression should be replaced
                auto it = std::ranges::find_if(
                    m_replacements, [&expr](ReplacementMapping const& mapping) {
                        return std::ranges::any_of(mapping.toReplace, [&expr](auto const& target) {
                            return Expression::identical(expr, target);
                        });
                    });

                if(it != m_replacements.end())
                    return it->canonical;

                // Otherwise, recursively process sub-expressions
                return std::visit(*this, *expr);
            }

        private:
            std::vector<ReplacementMapping> const& m_replacements;
        };

        /**
         * Visitor to apply expression replacements to all dimensions and operations in the graph
         */
        struct ReplaceInGraphVisitor
        {
            ReplaceInGraphVisitor(std::vector<ReplacementMapping> const& replacements)
                : m_exprVisitor{replacements}
            {
            }

            /**
             * Helper function that calls the expression visitor
             */
            Expression::ExpressionPtr replaceExpression(Expression::ExpressionPtr expr) const
            {
                return m_exprVisitor.call(expr);
            }

            template <CoordinateGraph::CCoordinateTransformEdge T>
            CoordinateGraph::Edge visitCoordinateEdge(int tag, T const& edge)
            {
                return edge;
            }

            template <CoordinateGraph::CDataFlowEdge T>
            CoordinateGraph::Edge visitCoordinateEdge(int tag, T const& edge)
            {
                return edge;
            }

            template <CoordinateGraph::CDimension T>
            CoordinateGraph::Dimension visitDimension(int tag, T const& dim)
            {
                auto d   = dim;
                d.size   = replaceExpression(dim.size);
                d.stride = replaceExpression(dim.stride);
                d.offset = replaceExpression(dim.offset);
                return d;
            }

            template <ControlGraph::COperation T>
            ControlGraph::Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::Assign const& op)
            {
                auto newOp       = op;
                newOp.expression = replaceExpression(op.expression);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::ConditionalOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceExpression(op.condition);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::AssertOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceExpression(op.condition);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::ForLoopOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceExpression(op.condition);
                return newOp;
            }

        private:
            ReplaceExprWithExprVisitor m_exprVisitor;
        };

        KernelGraph IdentifyParallelDimensions::apply(KernelGraph const& original)
        {
            auto parallelDims = mergeSets(identifyParallelDimensionSets(original));
            if(parallelDims.empty())
            {
                Log::debug("IdentifyParallelDimensions: No parallel dimensions found");
                return original;
            }

            auto copy = original;

            // Canonical expression to vector of expressions to replace
            std::vector<ReplacementMapping> replacements;

            for(auto const& dimSet : parallelDims)
            {
                AssertFatal(dimSet.size() > 1,
                            "Parallel dimension sets should have at least 2 dimensions");

                Expression::ExpressionPtr canonicalSize = nullptr;

                // Find canonical size (first non-null)
                for(int dim : dimSet)
                {
                    auto const& subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(subDim->size)
                    {
                        canonicalSize = subDim->size;
                        break;
                    }
                }

                AssertFatal(canonicalSize, "Replacement sets must have a canonical size");

                std::vector<Expression::ExpressionPtr> toReplace;

                // Collect expressions to replace with canonical
                for(int dim : dimSet)
                {
                    auto subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(Expression::identical(subDim->size, canonicalSize))
                        continue;

                    if(subDim->size)
                        toReplace.push_back(subDim->size);
                }

                if(!toReplace.empty())
                    replacements.push_back({canonicalSize, std::move(toReplace)});
            }

            size_t totalReplacements = 0;
            for(auto const& [canonical, toReplace] : replacements)
                totalReplacements += toReplace.size();

            Log::debug("IdentifyParallelDimensions: Replacing {} expression(s) with {} canonical "
                       "expression(s)",
                       totalReplacements,
                       replacements.size());

            auto visitor = ReplaceInGraphVisitor{replacements};
            copy         = rewriteDimensions(copy, visitor);

            return copy;
        }
    }
}

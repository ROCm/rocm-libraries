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

#include <rocRoller/KernelGraph/Reindexer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;

        /**
         * @brief Reindex coordinate tags in an Expression according to a reindexer.
         *
         * Expressions can contain references to tags within a coordinate graph.  For example, a
         * DataFlowTag expression represents the value of a coordinate within a coordinate graph,
         * and hence stores an explicit tag.
         *
         * After re-writing a coordinate graph, tag references need to be updated in all
         * expressions.
         */
        struct ReindexExpressionVisitor
        {
            ReindexExpressionVisitor(GraphReindexer const& reindexer)
                : m_reindexer(reindexer)
            {
            }

            template <Expression::CUnary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CBinary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            Expression::ExpressionPtr operator()(Expression::ScaledMatrixMultiply const& expr) const
            {
                Expression::ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CTernary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            Expression::ExpressionPtr operator()(CommandArgumentPtr const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr operator()(Expression::DataFlowTag const& expr) const
            {
                if(m_reindexer.coordinates.count(expr.tag) > 0)
                {
                    return std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                        m_reindexer.coordinates.at(expr.tag), expr.regType, expr.varType});
                }
                return std::make_shared<Expression::Expression>(expr);
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

                return std::visit(*this, *expr);
            }

        private:
            GraphReindexer const& m_reindexer;
        };

        Expression::ExpressionPtr reindexExpression(Expression::ExpressionPtr expr,
                                                    GraphReindexer const&     reindexer)
        {
            ReindexExpressionVisitor visitor(reindexer);
            return visitor.call(expr);
        }

        /**
         * @brief Reindex expressions inside the control nodes.
         */
#define MAKE_OPERATION_VISITOR(CLS) \
    void visitOperation(KernelGraph& graph, int tag, CLS const& dst) {}
        struct ReindexVisitor
        {
            ReindexVisitor(GraphReindexer const& reindexer)
                : m_reindexer{reindexer}
            {
            }

            void visitOperation(KernelGraph& graph, int tag, Assign const& assign)
            {
                auto newOp = Assign{assign.regType,
                                    reindexExpression(assign.expression, m_reindexer),
                                    assign.valueCount,
                                    assign.variableType};
                graph.control.setElement(tag, newOp);
            }

            void visitOperation(KernelGraph& graph, int tag, ConditionalOp const& op)
            {
                auto newOp
                    = ConditionalOp{reindexExpression(op.condition, m_reindexer), op.conditionName};
                graph.control.setElement(tag, newOp);
            }

            void visitOperation(KernelGraph& graph, int tag, AssertOp const& op)
            {
                auto newOp = AssertOp{op.assertName, reindexExpression(op.condition, m_reindexer)};
                graph.control.setElement(tag, newOp);
            }

            void visitOperation(KernelGraph& graph, int tag, ForLoopOp const& op)
            {
                auto newOp = ForLoopOp{reindexExpression(op.condition, m_reindexer), op.loopName};
                graph.control.setElement(tag, newOp);
            }

            void visitOperation(KernelGraph& graph, int tag, SetCoordinate const& op)
            {
                auto newOp = SetCoordinate(reindexExpression(op.value, m_reindexer));
                graph.control.setElement(tag, newOp);
            }

            MAKE_OPERATION_VISITOR(Barrier);
            MAKE_OPERATION_VISITOR(ComputeIndex);
            MAKE_OPERATION_VISITOR(Deallocate);
            MAKE_OPERATION_VISITOR(DoWhileOp);
            MAKE_OPERATION_VISITOR(Exchange);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadTileDirect2LDS);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(LoadSGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(NOP);
            MAKE_OPERATION_VISITOR(Block);
            MAKE_OPERATION_VISITOR(Scope);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(StoreSGPR);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(UnrollOp);
            MAKE_OPERATION_VISITOR(WaitZero);
            MAKE_OPERATION_VISITOR(SeedPRNG);

            void reindex(KernelGraph& graph, int tag)
            {
                auto element = graph.control.getElement(tag);
                if(std::holds_alternative<Operation>(element))
                {
                    auto node = std::get<Operation>(element);
                    std::visit([&](auto arg) { visitOperation(graph, tag, arg); }, node);
                }
            }

        private:
            GraphReindexer const& m_reindexer;
        };
#undef MAKE_OPERATION_VISITOR

        void reindexExpressions(KernelGraph& graph, int tag, GraphReindexer const& reindexer)
        {
            ReindexVisitor visitor(reindexer);
            visitor.reindex(graph, tag);
        }
    }
}

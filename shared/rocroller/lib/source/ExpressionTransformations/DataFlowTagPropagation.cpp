// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct DataFlowTagPropagationVisitor
        {
            DataFlowTagPropagationVisitor(RegisterTagManager const& tagManager)
                : m_tagManager(tagManager)
            {
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr)
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(DataFlowTag const& expr)
            {
                if(m_tagManager.hasExpression(expr.tag))
                {
                    auto [tagExpr, _ignore] = m_tagManager.getExpression(expr.tag);
                    return call(tagExpr);
                }
                else
                {
                    AssertFatal(m_tagManager.hasRegister(expr.tag), ShowValue(expr.tag));
                    return std::make_shared<Expression>(m_tagManager.getRegister(expr.tag));
                }
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr)
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr)
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            RegisterTagManager const& m_tagManager;
        };

        ExpressionPtr dataFlowTagPropagation(ExpressionPtr             expr,
                                             RegisterTagManager const& tagManager)
        {
            auto visitor = DataFlowTagPropagationVisitor(tagManager);
            return visitor.call(expr);
        }

        ExpressionPtr dataFlowTagPropagation(ExpressionPtr expr, ContextPtr context)
        {
            return dataFlowTagPropagation(expr, *context->registerTagManager());
        }
    }
}

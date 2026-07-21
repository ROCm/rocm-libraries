// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct PositionalArgumentPropagationVisitor
        {
            PositionalArgumentPropagationVisitor(ContextPtr                 context,
                                                 std::vector<ExpressionPtr> arguments)
                : m_context(context)
                , m_arguments(arguments)
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
                DataFlowTag cpy = expr;
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(PositionalArgument const& expr)
            {
                AssertFatal(expr.slot >= 0 && expr.slot < m_arguments.size(), ShowValue(expr.slot));

                auto replacement = m_arguments[expr.slot];

                auto [regType, varType, valueCount] = resultType(replacement);

                if(regType != expr.regType || varType != expr.varType)
                    Log::debug("Type mismatch for PositionalArgument({}):\n"
                               "expecting ({}, {})\n"
                               "with argument {}\n"
                               "which is ({},{}).",
                               expr.slot,
                               toString(expr.regType),
                               toString(expr.varType),
                               toString(replacement),
                               toString(regType),
                               toString(varType));

                return call(replacement);
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
            ContextPtr                 m_context;
            std::vector<ExpressionPtr> m_arguments;
        };

        ExpressionPtr positionalArgumentPropagation(ExpressionPtr              expr,
                                                    std::vector<ExpressionPtr> arguments)
        {
            auto visitor = PositionalArgumentPropagationVisitor(nullptr, arguments);
            return visitor.call(expr);
        }

    }
}

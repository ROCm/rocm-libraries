// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct LowerUnsignedArithmeticShiftRExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
            }

            ExpressionPtr operator()(ArithmeticShiftR const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                auto const resultType = resultVariableType(lhs);
                if(not DataTypeInfo::Get(resultType).isSigned)
                {
                    return std::make_shared<Expression>(LogicalShiftR({lhs, rhs}));
                }

                return std::make_shared<Expression>(ArithmeticShiftR({lhs, rhs}));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        ExpressionPtr lowerUnsignedArithmeticShiftR(ExpressionPtr expr)
        {
            auto visitor = LowerUnsignedArithmeticShiftRExpressionVisitor();
            return visitor.call(expr);
        }
    }
}

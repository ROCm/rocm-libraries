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

                auto [regType, varType] = resultType(replacement);

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

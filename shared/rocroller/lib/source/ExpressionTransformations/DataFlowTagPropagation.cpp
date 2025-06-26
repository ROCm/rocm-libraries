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
        struct DataFlowTagPropagationVisitor
        {
            DataFlowTagPropagationVisitor(RegisterTagManager const& tagManager)
                : m_tagManager(tagManager)
            {
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr)
            {
                ScaledMatrixMultiply cpy = expr;
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
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
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
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
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
                return std::make_shared<Expression>(cpy);
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
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

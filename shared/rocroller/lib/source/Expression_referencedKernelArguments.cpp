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

#include <variant>

#include <rocRoller/Expression.hpp>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct ExpressionKernelArgumentsVisitor
        {
            void call(ExpressionPtr const& expr)
            {
                if(expr)
                    call(*expr);
            }

            void call(Expression const& expr)
            {
                std::visit(*this, expr);
            }

            void operator()(CUnary auto const& expr)
            {
                call(expr.arg);
            }

            void operator()(CBinary auto const& expr)
            {
                call(expr.lhs);
                call(expr.rhs);
            }

            void operator()(CTernary auto const& expr)
            {
                call(expr.lhs);
                call(expr.r1hs);
                call(expr.r2hs);
            }

            void operator()(ScaledMatrixMultiply const& expr)
            {
                call(expr.matA);
                call(expr.matB);
                call(expr.matC);
                call(expr.scaleA);
                call(expr.scaleB);
            }

            void operator()(AssemblyKernelArgumentPtr const& expr)
            {
                m_referencedArgs.insert(expr->name);
            }

            void operator()(DataFlowTag const& expr)
            {
                if(m_tagManager.hasExpression(expr.tag))
                {
                    auto [subExpr, _] = m_tagManager.getExpression(expr.tag);
                    call(subExpr);
                }
            }

            void operator()(CValue auto const& expr) {}

            RegisterTagManager const& m_tagManager;

            std::unordered_set<std::string> m_referencedArgs;
        };

        std::unordered_set<std::string>
            referencedKernelArguments(Expression const& expr, RegisterTagManager const& tagManager)
        {
            ExpressionKernelArgumentsVisitor visitor{tagManager};
            visitor.call(expr);

            return std::move(visitor.m_referencedArgs);
        }

        std::unordered_set<std::string>
            referencedKernelArguments(ExpressionPtr const&      expr,
                                      RegisterTagManager const& tagManager)
        {
            if(expr)
                return referencedKernelArguments(*expr, tagManager);

            return {};
        }

        std::unordered_set<std::string> referencedKernelArguments(Expression const& expr)
        {
            RegisterTagManager tagManager(nullptr);

            return referencedKernelArguments(expr, tagManager);
        }

        std::unordered_set<std::string> referencedKernelArguments(ExpressionPtr const& expr)
        {
            if(expr)
                return referencedKernelArguments(*expr);

            return {};
        }
    }
}

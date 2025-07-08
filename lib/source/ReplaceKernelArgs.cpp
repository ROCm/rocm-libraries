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

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ReplaceKernelArgs.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct ReplaceKernelArgsVisitor
        {
            ReplaceKernelArgsVisitor(ContextPtr const& context)
                : m_context(context)
            {
            }

            Generator<Instruction> operator()(AssemblyKernelArgumentPtr const& expr)
            {
                auto iter = m_values.find(expr->name);
                if(iter != m_values.end())
                {
                    m_lastResult = iter->second;
                    co_return;
                }

                Register::ValuePtr reg;
                co_yield m_context->argLoader()->getValue(expr->name, reg);

                auto exp             = std::make_shared<Expression>(reg);
                m_values[expr->name] = exp;
                m_lastResult         = exp;
            }

            template <CUnary T>
            Generator<Instruction> operator()(T expr)
            {
                co_yield call(expr.arg);
                m_lastResult = std::make_shared<Expression>(expr);
            }

            template <CBinary T>
            Generator<Instruction> operator()(T expr)
            {
                co_yield call(expr.lhs);
                co_yield call(expr.rhs);
                m_lastResult = std::make_shared<Expression>(expr);
            }

            template <CTernary T>
            Generator<Instruction> operator()(T expr)
            {
                co_yield call(expr.lhs);
                co_yield call(expr.r1hs);
                co_yield call(expr.r2hs);
                m_lastResult = std::make_shared<Expression>(expr);
            }

            Generator<Instruction> operator()(ScaledMatrixMultiply expr)
            {
                co_yield call(expr.matA);
                co_yield call(expr.matB);
                co_yield call(expr.matC);
                co_yield call(expr.scaleA);
                co_yield call(expr.scaleB);
                m_lastResult = std::make_shared<Expression>(expr);
            }

            template <CIsAnyOf<CommandArgumentValue,
                               CommandArgumentPtr,
                               PositionalArgument,
                               Register::ValuePtr,
                               DataFlowTag,
                               WaveTilePtr> T>
            Generator<Instruction> operator()(T expr)
            {
                m_lastResult = std::make_shared<Expression>(expr);
                co_return;
            }

            Generator<Instruction> call(ExpressionPtr& dst, ExpressionPtr const& src)
            {
                co_yield std::visit(*this, *src);
                dst = m_lastResult;
            }

            Generator<Instruction> call(ExpressionPtr& inout)
            {
                co_yield call(inout, inout);
            }

        private:
            ContextPtr                           m_context;
            std::map<std::string, ExpressionPtr> m_values;
            ExpressionPtr                        m_lastResult = nullptr;
        };

        Generator<Instruction> replaceKernelArgs(ContextPtr const&    context,
                                                 ExpressionPtr&       dst,
                                                 ExpressionPtr const& src)
        {
            ReplaceKernelArgsVisitor visitor(context);
            co_yield visitor.call(dst, src);
        }
    }
}

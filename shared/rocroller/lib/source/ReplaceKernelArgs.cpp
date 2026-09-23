// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
                auto iter = m_values.find(expr->getName());
                if(iter != m_values.end())
                {
                    m_lastResult = iter->second;
                    co_return;
                }

                Register::ValuePtr reg;
                co_yield m_context->argLoader()->getValue(expr->getName(), reg);

                auto exp                  = std::make_shared<Expression>(reg);
                m_values[expr->getName()] = exp;
                m_lastResult              = exp;
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

            template <CNary T>
            Generator<Instruction> operator()(T expr)
            {
                for(auto&& operand : expr.operands)
                {
                    co_yield call(operand);
                }
                m_lastResult = std::make_shared<Expression>(std::move(expr));
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

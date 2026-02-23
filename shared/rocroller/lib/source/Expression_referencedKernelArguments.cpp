// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

            void operator()(CNary auto const& expr)
            {
                std::ranges::for_each(expr.operands, [this](auto const& op) { call(op); });
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
                m_referencedArgs.insert(expr->getName());
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

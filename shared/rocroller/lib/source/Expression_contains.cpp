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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        template <CExpression T>
        struct ContainsVisitor
        {
            bool operator()(T const& expr)
            {
                return true;
            }

            template <CUnary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.arg);
            }

            template <CBinary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.lhs) || call(expr.rhs);
            }

            template <CTernary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.lhs) || call(expr.r1hs) || call(expr.r2hs);
            }

            template <std::same_as<ScaledMatrixMultiply> U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.matA) || call(expr.matB) || call(expr.matC) || call(expr.scaleA)
                       || call(expr.scaleB);
            }

            template <CValue U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return false;
            }

            bool call(Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            bool call(ExpressionPtr const& expr)
            {
                if(!expr)
                    return false;

                return call(*expr);
            }
        };

        template <CExpression T>
        __attribute__((noinline)) bool contains(Expression const& expr)
        {
            ContainsVisitor<T> v;
            return v.call(expr);
        }

        template <CExpression T>
        __attribute__((noinline)) bool contains(ExpressionPtr expr)
        {
            AssertFatal(expr != nullptr);

            return contains<T>(*expr);
        }

        /**
         * Force instantiation of contains() for every type of expression, so
         * that it can be implemented in the .cpp file.
         */
        struct ContainsInstantiateVisitor
        {
            ExpressionPtr expr;

            template <CExpression T>
            bool operator()(T const& exprType)
            {
                return contains<T>(expr);
            }
        };

        bool containsType(ExpressionPtr exprType, ExpressionPtr expr)
        {
            AssertFatal(exprType != nullptr);

            ContainsInstantiateVisitor v{expr};

            return std::visit(v, *exprType);
        }

        struct ContainsSubExpressionVisitor
        {
            ContainsSubExpressionVisitor(Expression const& expr)
                : subExpr(expr)
            {
            }

            ContainsSubExpressionVisitor(ExpressionPtr const& exprPtr)
                : subExpr(*exprPtr)
            {
            }

            template <CUnary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.arg);
            }

            template <CBinary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.lhs) || call(expr.rhs);
            }

            template <CTernary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.lhs) || call(expr.r1hs)
                       || call(expr.r2hs);
            }

            template <CValue U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr);
            }

            bool operator()(ScaledMatrixMultiply const& expr) const
            {
                return identical(expr, subExpr) || call(expr.matA) || call(expr.matB)
                       || call(expr.matC) || call(expr.scaleA) || call(expr.scaleB);
            }

            bool call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            bool call(ExpressionPtr const& expr) const
            {
                if(!expr)
                    return false;

                return call(*expr);
            }

            Expression const& subExpr;
        };

        bool containsSubExpression(ExpressionPtr const& expr, ExpressionPtr const& subExpr)
        {
            auto visitor = ContainsSubExpressionVisitor(subExpr);
            return visitor.call(expr);
        }

        bool containsSubExpression(Expression const& expr, Expression const& subExpr)
        {
            auto visitor = ContainsSubExpressionVisitor(subExpr);
            return visitor.call(expr);
        }
    }
}

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

#include <bit>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Fast Multiplication
         *
         * Attempt to replace multiplication operations found within an expression with faster
         * operations.
         */

        struct MultiplicationByConstant
        {
            DataType      resultType;
            ExpressionPtr lhs = nullptr;

            // Fast Multiplication for when rhs is power of two
            template <typename T>
            requires(std::integral<T> && !std::same_as<bool, T>) ExpressionPtr operator()(T rhs)
            {
                if(resultVariableType(lhs) != resultType)
                {
                    lhs = convert(resultType, lhs);
                }

                if(rhs == 0)
                {
                    return literal(0);
                }
                else if(rhs == 1)
                {
                    return lhs;
                }
                // Power of 2 Multiplication
                else if(rhs > 0 && std::has_single_bit(cast_to_unsigned(rhs)))
                {
                    auto rhs_exp
                        = literal(cast_to_unsigned(std::countr_zero(cast_to_unsigned(rhs))));

                    return lhs << rhs_exp;
                }

                return nullptr;
            }

            // If the rhs is not an integer, return a nullptr to indicate we can't optimize.
            template <typename T>
            requires(!std::integral<T> || std::same_as<bool, T>) ExpressionPtr operator()(T rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        struct FastMultiplicationExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
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

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
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
            ExpressionPtr operator()(Expr const& expr) const
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

            ExpressionPtr operator()(Multiply const& expr) const
            {
                auto origResultType = resultVariableType(expr);
                AssertFatal(!origResultType.isPointer(), ShowValue(origResultType));

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                auto mulByConst = MultiplicationByConstant{origResultType.dataType};

                if(evaluationTimes(rhs)[EvaluationTime::Translate])
                {
                    auto rhs_val = evaluate(rhs);
                    auto rv      = mulByConst.call(lhs, rhs_val);
                    if(rv != nullptr)
                    {
                        copyComment(rv, expr);
                        return rv;
                    }
                }

                if(evaluationTimes(lhs)[EvaluationTime::Translate])
                {
                    auto lhs_val = evaluate(lhs);
                    // lhs becomes rhs because visitor checks rhs for optimization
                    auto rv = mulByConst.call(rhs, lhs_val);
                    if(rv != nullptr)
                    {
                        copyComment(rv, expr);
                        return rv;
                    }
                }

                return std::make_shared<Expression>(Multiply({lhs, rhs, expr.comment}));
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

        /**
         * Attempts to use fastMultiplication for all of the multiplications within an Expression.
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr)
        {
            auto visitor = FastMultiplicationExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

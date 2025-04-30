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
        struct GetShiftInfo
        {
            struct Result
            {
                bool          isShift     = false;
                bool          isSigned    = false;
                bool          shiftLeft   = false;
                ExpressionPtr value       = nullptr;
                ExpressionPtr shiftAmount = nullptr;
            };

            Result operator()(auto const& expr) const
            {
                return {};
            }

            template <CShift Shift>
            Result operator()(Shift const& expr) const
            {
                bool isSigned = std::same_as<Shift, ArithmeticShiftR> //
                                && DataTypeInfo::Get(resultVariableType(expr.lhs)).isSigned;

                return {true, isSigned, std::same_as<Shift, ShiftL>, expr.lhs, expr.rhs};
            }

            Result call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }
        };

        struct CombineShiftsExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;

                cpy.arg = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            requires(!CShift<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;

                cpy.lhs = call(expr.lhs);
                cpy.rhs = call(expr.rhs);
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

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                auto cpy = expr;

                cpy.matA   = call(expr.matA);
                cpy.matB   = call(expr.matB);
                cpy.matC   = call(expr.matC);
                cpy.scaleA = call(expr.scaleA);
                cpy.scaleB = call(expr.scaleB);

                return std::make_shared<Expression>(cpy);
            }

            template <CShift Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                ExpressionPtr rv;

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                auto [_, isSigned, isLeft, __, shiftAmount] = GetShiftInfo().call(Expr{lhs, rhs});

                auto [lhsIsShift, lhsIsSigned, lhsIsLeft, lhsValue, lhsAmount]
                    = GetShiftInfo().call(*lhs);

                if(lhsIsShift)
                {
                    if(!isSigned && (isLeft != lhsIsLeft))
                    {
                        // Left then right. AND-off MSBs, if the shifts are the same.
                        // Or, Right then left. AND-off LSBs, if the shifts are the same.

                        auto sameShift = (shiftAmount == lhsAmount);
                        if(canEvaluateTo(true, sameShift))
                        {
                            // If we get here then:
                            // - Both shifts can be evaluated at translate time.
                            // - Both shifts are equal to each other.

                            int bits = getUnsignedInt(evaluate(shiftAmount));

                            auto lhsValueType = resultVariableType(lhsValue);
                            if(lhsValueType.getElementSize() == 4)
                            {
                                auto mask = (~(0u));
                                if(isLeft)
                                    mask <<= bits;
                                else
                                    mask >>= bits;
                                static_assert(sizeof(mask) == 4);
                                rv = lhsValue & literal(mask, lhsValueType);
                            }
                            else if(lhsValueType.getElementSize() == 8)
                            {
                                auto mask = (~(0ul));
                                if(isLeft)
                                    mask <<= bits;
                                else
                                    mask >>= bits;
                                static_assert(sizeof(mask) == 8);
                                rv = lhsValue & literal(mask, lhsValueType);
                            }
                            // Else: leave the original expression.
                        }
                    }
                }

                if(rv != nullptr)
                {
                    copyComment(rv, lhs);
                    copyComment(rv, expr);
                    return rv;
                }

                return std::make_shared<Expression>(Expr{lhs, rhs, expr.comment});
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

                auto resultVarType = resultVariableType(expr);

                auto rv = std::visit(*this, *expr);

                if(resultVariableType(rv) != resultVarType)
                    return convert(resultVarType, rv);

                return rv;
            }
        };

        /**
         * Attempts to use combineShifts for all of the shifts within an Expression.
         */
        ExpressionPtr combineShifts(ExpressionPtr expr)
        {
            auto visitor = CombineShiftsExpressionVisitor();
            auto rv      = visitor.call(expr);

            AssertFatal(resultVariableType(expr) == resultVariableType(rv),
                        "Transformation should not alter result type.",
                        ShowValue(expr),
                        ShowValue(rv));

            return rv;
        }

    }
}

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

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        struct FuseExpressionVisitor
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
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Add const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                if(auto const* shift = std::get_if<ShiftL>(lhs.get()))
                {
                    bool eval_shift_rhs = evaluationTimes(shift->rhs)[EvaluationTime::Translate];
                    if(eval_shift_rhs)
                    {
                        auto comment = shift->comment + expr.comment;
                        auto rv      = shiftLAdd(shift->lhs, shift->rhs, rhs);
                        setComment(rv, comment);
                        return rv;
                    }
                }

                if(auto const* multiply = std::get_if<Multiply>(lhs.get()))
                {
                    auto comment = multiply->comment + expr.comment;

                    auto rv = multiplyAdd(multiply->lhs, multiply->rhs, rhs);
                    setComment(rv, comment);
                    return rv;
                }

                return std::make_shared<Expression>(Add({lhs, rhs, expr.comment}));
            }

            ExpressionPtr operator()(ShiftL const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                if(auto const* add = std::get_if<Add>(lhs.get()))
                {
                    bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];
                    if(eval_rhs)
                    {
                        auto add     = std::get<Add>(*lhs);
                        auto comment = add.comment + expr.comment;
                        return std::make_shared<Expression>(
                            AddShiftL{add.lhs, add.rhs, rhs, comment});
                    }
                }

                return std::make_shared<Expression>(expr);
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                AssertFatal(expr != nullptr, "Found nullptr in expression");
                return std::visit(*this, *expr);
            }
        };

        ExpressionPtr fuseTernary(ExpressionPtr expr)
        {
            auto visitor = FuseExpressionVisitor();
            return visitor.call(expr);
        }
    }
}

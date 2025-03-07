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
        /**
         * Simplify trivial arithmetic expressions involving translation time constants.
         *
         * Simplifications:
         * - Add two integers, or add 0
         * - Multiply two integers, multiply by 0, multiply by 1
         * - Divide by 1
         * - Modulo by 1
         * - Shift by 0
         */

        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        template <typename T>
        concept CBoolean = std::same_as<bool, T>;

        template <typename T>
        struct SimplifyByConstant
        {
            VariableType resultVarType;

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                return nullptr;
            }
        };

        template <typename T>
        struct SimplifyByConstantLHS
        {
            VariableType resultVarType;

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<Add>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <CShift ShiftType>
        struct SimplifyByConstant<ShiftType>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<BitwiseAnd>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<LogicalAnd>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == false)
                    return literal(false);
                if(rhs == true)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<LogicalOr>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs = nullptr;

            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == true)
                    return literal(true);
                if(rhs == false)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<Multiply>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return literal(0, resultVarType);
                if(rhs == 1)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<Divide>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstantLHS<Divide>
        {
            VariableType  resultVarType;
            ExpressionPtr rhs;

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs_)
            {
                rhs = rhs_;
                return visit(*this, lhs);
            }
        };

        template <>
        struct SimplifyByConstant<Modulo>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstantLHS<Modulo>
        {
            VariableType  resultVarType;
            ExpressionPtr rhs;

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs_)
            {
                rhs = rhs_;
                return visit(*this, lhs);
            }
        };

        struct SimplifyExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                if constexpr(Expr::Type == Category::Conversion)
                {
                    if(expr.arg)
                    {
                        if(resultVariableType(expr) == resultVariableType(expr.arg))
                            return call(expr.arg);
                    }
                }

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
                auto resultVarType = resultVariableType(expr);

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyByConstant<Expr>{resultVarType};

                ExpressionPtr rv = nullptr;

                if(eval_lhs && eval_rhs)
                {
                    rv = literal(evaluate(Expr({lhs, rhs})));
                }
                else if(CCommutativeBinary<Expr> && eval_lhs)
                {
                    rv = simplifier.call(rhs, evaluate(lhs));
                }
                else if(eval_rhs)
                {
                    rv = simplifier.call(lhs, evaluate(rhs));
                }
                else if(!CCommutativeBinary<Expr> && eval_lhs)
                {
                    auto simplifierLHS = SimplifyByConstantLHS<Expr>{resultVarType};

                    rv = simplifierLHS.call(evaluate(lhs), rhs);
                }

                if(rv != nullptr)
                {
                    if(resultVariableType(rv) != resultVarType)
                    {
                        AssertFatal(!resultVarType.isPointer(),
                                    ShowValue(expr),
                                    ShowValue(rv),
                                    ShowValue(resultVarType));
                        rv = convert(resultVarType.dataType, rv);
                    }

                    copyComment(rv, expr);
                    return rv;
                }

                return std::make_shared<Expression>(Expr({lhs, rhs, expr.comment}));
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

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                auto rv = std::visit(*this, *expr);
                return rv;
            }
        };

        ExpressionPtr simplify(ExpressionPtr expr)
        {
            auto visitor = SimplifyExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

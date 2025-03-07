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
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        template <typename T>
        concept CConstant = CIntegral<T> || std::floating_point<T>;

        template <CAssociativeBinary OP>
        struct AssociativeBinary
        {
            template <typename RHS>
            ExpressionPtr operator()(RHS rhs)
            {
                if(std::holds_alternative<OP>(*m_lhs))
                {
                    auto lhs_op = std::get<OP>(*m_lhs);

                    bool eval_lhs = evaluationTimes(lhs_op.lhs)[EvaluationTime::Translate];
                    bool eval_rhs = evaluationTimes(lhs_op.rhs)[EvaluationTime::Translate];

                    OP operation;
                    if(CCommutativeBinary<OP> && eval_lhs)
                    {
                        operation.lhs
                            = simplify(std::make_shared<Expression>(OP{lhs_op.lhs, literal(rhs)}));
                        operation.rhs = lhs_op.rhs;
                    }
                    else
                    {
                        operation.lhs = lhs_op.lhs;
                        operation.rhs
                            = simplify(std::make_shared<Expression>(OP{lhs_op.rhs, literal(rhs)}));
                    }

                    return std::make_shared<Expression>(operation);
                }
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

        private:
            ExpressionPtr m_lhs;
        };

        template <CShift OP>
        struct CollectedShift
        {
            template <typename RHS>
            ExpressionPtr operator()(RHS rhs)
            {
                if(std::holds_alternative<OP>(*m_lhs))
                {
                    auto lhs_op = std::get<OP>(*m_lhs);

                    bool eval_rhs = evaluationTimes(lhs_op.rhs)[EvaluationTime::Translate];

                    if(eval_rhs)
                    {
                        OP operation;
                        operation.lhs = lhs_op.lhs;
                        operation.rhs
                            = simplify(std::make_shared<Expression>(Add{lhs_op.rhs, literal(rhs)}));

                        return std::make_shared<Expression>(operation);
                    }
                }
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

        private:
            ExpressionPtr m_lhs;
        };

        struct AssociativeExpressionVisitor
        {
            template <CAssociativeBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto associativeBinary = AssociativeBinary<Expr>();

                ExpressionPtr rv;

                if(eval_lhs && eval_rhs)
                {
                    rv = literal(evaluate(std::make_shared<Expression>(Expr{lhs, rhs})));
                }
                else if(CCommutativeBinary<Expr> && eval_lhs)
                {
                    rv = associativeBinary.call(rhs, evaluate(lhs));
                }
                else if(eval_rhs)
                {
                    rv = associativeBinary.call(lhs, evaluate(rhs));
                }

                if(rv != nullptr)
                {
                    copyComment(rv, expr);
                    return rv;
                }

                return std::make_shared<Expression>(Expr({lhs, rhs, expr.comment}));
            }

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
            requires(!CAssociativeBinary<Expr> && !CShift<Expr>) ExpressionPtr
                operator()(Expr const& expr) const
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

            template <CShift Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto collectedShift = CollectedShift<Expr>();

                ExpressionPtr rv;

                if(eval_lhs && eval_rhs)
                {
                    rv = literal(evaluate(std::make_shared<Expression>(Expr{lhs, rhs})));
                }
                else if(eval_rhs)
                {
                    rv = collectedShift.call(lhs, evaluate(rhs));
                }

                if(rv != nullptr)
                {
                    copyComment(rv, expr);
                    return rv;
                }

                return std::make_shared<Expression>(Expr({lhs, rhs, expr.comment}));
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

        ExpressionPtr fuseAssociative(ExpressionPtr expr)
        {
            auto visitor = AssociativeExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

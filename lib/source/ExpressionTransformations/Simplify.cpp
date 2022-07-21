#include <rocRoller/AssemblyKernel.hpp>
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
         * Simplify trivial arithmetic expressions involving translation time constants.
         *
         * Simplifications:
         * - Add two integers, or add 0
         * - Multiply two integers, multiply by 0, multiply by 1
         * - Divide by 1
         * - Modulo by 1
         */

        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        struct SimplifyAdditionByConstant
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(rhs + lhs);
            }

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                if(lhs == 0)
                    return literal(rhs);
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return (*this)(rhs, lhs);
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return m_lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct SimplifyMultiplicationByConstant
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(rhs * lhs);
            }

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                if(lhs == 0)
                    return literal(0);
                if(lhs == 1)
                    return literal(rhs);
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return (*this)(rhs, lhs);
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return literal(0);
                if(rhs == 1)
                    return m_lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct SimplifyDivideByConstant
        {
            ExpressionPtr m_lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return m_lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct SimplifyModuloByConstant
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(lhs % rhs);
            }

            template <typename LHS, typename RHS>
            requires(!(CIntegral<LHS> && CIntegral<RHS>)) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return literal(0);
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct SimplifyExpressionVisitor
        {
            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({(*this)(expr.lhs), (*this)(expr.rhs)}));
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(
                    Expr({(*this)(expr.lhs), (*this)(expr.r1hs), (*this)(expr.r2hs)}));
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({(*this)(expr.arg)}));
            }

            ExpressionPtr operator()(Add const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyAdditionByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = std::visit(simplifier, evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = simplifier(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = simplifier(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Add({lhs, rhs}));
            }

            ExpressionPtr operator()(Multiply const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyMultiplicationByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = std::visit(simplifier, evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = simplifier(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = simplifier(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Multiply({lhs, rhs}));
            }

            ExpressionPtr operator()(Divide const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                if(evaluationTimes(rhs)[EvaluationTime::Translate])
                {
                    auto rhs_val = evaluate(rhs);
                    auto rv      = SimplifyDivideByConstant()(lhs, rhs_val);
                    if(rv != nullptr)
                        return rv;
                }

                return std::make_shared<Expression>(Divide({lhs, rhs}));
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyModuloByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = std::visit(simplifier, evaluate(lhs), evaluate(rhs));
                else if(eval_rhs)
                    rv = simplifier(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Modulo({lhs, rhs}));
            }

            // TODO add to binary
            ExpressionPtr operator()(MatrixMultiply const& expr) const
            {
                return std::make_shared<Expression>(MatrixMultiply(expr));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        /**
         * Attempts to use simplify an Expression.
         */
        ExpressionPtr simplify(ExpressionPtr expr)
        {
            auto visitor = SimplifyExpressionVisitor();
            return visitor(expr);
        }

    }
}

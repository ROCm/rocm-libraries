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
                return call(rhs, lhs);
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
            }
        };

        struct SimplifyShiftLByConstant
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(lhs << rhs);
            }

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                if(rhs == 0)
                    return literal(lhs);
                return nullptr;
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
            }
        };

        struct SimplifySignedShiftRByConstant
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(lhs >> rhs);
            }

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS> && !CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                if(rhs == 0)
                    return literal(lhs);
                return nullptr;
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
            }
        };

        struct SimplifyBitwiseAnd
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS>&& CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return literal(lhs & rhs);
            }

            template <typename LHS, typename RHS>
            requires(CIntegral<LHS> && !CIntegral<RHS> && !CBinary<RHS>) ExpressionPtr
                operator()(LHS lhs, RHS rhs)
            {
                if(lhs == 0)
                {
                    return literal(0);
                }
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CIntegral<LHS> && CIntegral<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return call(rhs, lhs);
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
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
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
                return call(rhs, lhs);
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
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
                AssertFatal(rhs != 0, "Modulo operation by 0");
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

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ExpressionPtr call(CommandArgumentValue lhs, CommandArgumentValue rhs)
            {
                return visit(*this, lhs, rhs);
            }
        };

        struct SimplifyExpressionVisitor
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

            ExpressionPtr operator()(Add const& expr) const
            {
                static_assert(CCommutativeBinary<Add>);

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyAdditionByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = simplifier.call(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Add({lhs, rhs}));
            }

            ExpressionPtr operator()(ShiftL const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyShiftLByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(ShiftL({lhs, rhs}));
            }

            ExpressionPtr operator()(SignedShiftR const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifySignedShiftRByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(SignedShiftR({lhs, rhs}));
            }

            ExpressionPtr operator()(Multiply const& expr) const
            {
                static_assert(CCommutativeBinary<Multiply>);

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyMultiplicationByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = simplifier.call(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Multiply({lhs, rhs}));
            }

            ExpressionPtr operator()(Divide const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                if(evaluationTimes(rhs)[EvaluationTime::Translate])
                {
                    auto simplifier = SimplifyDivideByConstant();
                    auto rv         = simplifier.call(lhs, evaluate(rhs));
                    if(rv != nullptr)
                        return rv;
                }

                return std::make_shared<Expression>(Divide({lhs, rhs}));
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyModuloByConstant();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Modulo({lhs, rhs}));
            }

            ExpressionPtr operator()(BitwiseAnd const& expr) const
            {
                static_assert(CCommutativeBinary<BitwiseAnd>);

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyBitwiseAnd();

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs)
                    rv = simplifier.call(evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = simplifier.call(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = simplifier.call(lhs, evaluate(rhs));
                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(BitwiseAnd({lhs, rhs}));
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

        ExpressionPtr simplify(ExpressionPtr expr)
        {
            auto visitor = SimplifyExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

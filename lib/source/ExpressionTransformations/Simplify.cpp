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
            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                return nullptr;
            }
        };

        template <typename T>
        struct SimplifyByConstantLHS
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<Add>
        {
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

        private:
            ExpressionPtr m_lhs;
        };

        template <>
        struct SimplifyByConstantLHS<Add>
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <CShift ShiftType>
        struct SimplifyByConstant<ShiftType>
        {
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

        private:
            ExpressionPtr m_lhs;
        };

        template <CShift ShiftType>
        struct SimplifyByConstantLHS<ShiftType>
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<BitwiseAnd>
        {
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

        private:
            ExpressionPtr m_lhs;
        };

        template <>
        struct SimplifyByConstantLHS<BitwiseAnd>
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<LogicalAnd>
        {
            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == false)
                    return literal(false);
                if(rhs == true)
                    return m_lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
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

        template <>
        struct SimplifyByConstant<LogicalOr>
        {
            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == true)
                    return literal(true);
                if(rhs == false)
                    return m_lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
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

        template <>
        struct SimplifyByConstantLHS<LogicalAnd>
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<Multiply>
        {
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

        private:
            ExpressionPtr m_lhs;
        };

        template <>
        struct SimplifyByConstantLHS<Multiply>
        {
            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<Divide>
        {

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

        private:
            ExpressionPtr m_lhs;
        };

        template <>
        struct SimplifyByConstantLHS<Divide>
        {

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                m_rhs = rhs;
                return visit(*this, lhs);
            }

        private:
            ExpressionPtr m_rhs;
        };

        template <>
        struct SimplifyByConstant<Modulo>
        {

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

        private:
            ExpressionPtr m_lhs;
        };

        template <>
        struct SimplifyByConstantLHS<Modulo>
        {

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                m_rhs = rhs;
                return visit(*this, lhs);
            }

        private:
            ExpressionPtr m_rhs;
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
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyByConstant<Expr>();

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
                    auto simplifierLHS = SimplifyByConstantLHS<Expr>();

                    rv = simplifierLHS.call(evaluate(lhs), rhs);
                }

                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Expr({lhs, rhs, expr.comment}));
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

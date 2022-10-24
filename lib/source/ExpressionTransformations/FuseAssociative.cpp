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
        requires(CCommutativeBinary<OP>) struct AssociativeBinary
        {
            ExpressionPtr m_lhs;

            template <typename LHS, typename RHS>
            requires(CConstant<LHS>&& CConstant<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                ExpressionPtr operation
                    = std::make_shared<Expression>(OP{literal(lhs), literal(rhs)});
                return simplify(operation);
            }

            template <typename LHS>
            requires(CConstant<LHS>) ExpressionPtr operator()(LHS lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(CConstant<LHS> && !CConstant<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename LHS, typename RHS>
            requires(!CConstant<LHS> && CConstant<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return (*this)(rhs, lhs);
            }

            template <typename LHS, typename RHS>
            requires(!CConstant<LHS> && !CConstant<RHS>) ExpressionPtr operator()(LHS lhs, RHS rhs)
            {
                return nullptr;
            }

            template <typename RHS>
            ExpressionPtr operator()(RHS rhs)
            {
                if(std::holds_alternative<OP>(*m_lhs))
                {
                    auto lhs_op = std::get<OP>(*m_lhs);

                    ExpressionPtr simple
                        = std::make_shared<Expression>(OP{lhs_op.rhs, literal(rhs)});

                    OP operation;
                    operation.lhs = lhs_op.lhs;
                    operation.rhs = simplify(simple);
                    return std::make_shared<Expression>(operation);
                }
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct AssociativeExpressionVisitor
        {
            template <CAssociativeBinary Expr>
            requires(CCommutativeBinary<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto associativeBinary = AssociativeBinary<Expr>();

                ExpressionPtr rv;

                if(eval_lhs && eval_rhs)
                    rv = std::visit(associativeBinary, evaluate(lhs), evaluate(rhs));
                else if(eval_lhs)
                    rv = associativeBinary(rhs, evaluate(lhs));
                else if(eval_rhs)
                    rv = associativeBinary(lhs, evaluate(rhs));

                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Expr({lhs, rhs}));
            }

            template <CBinary Expr>
            requires(!CAssociativeBinary<Expr>) ExpressionPtr operator()(Expr const& expr) const
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

        ExpressionPtr fuseAssociative(ExpressionPtr expr)
        {
            auto visitor = AssociativeExpressionVisitor();
            return visitor(expr);
        }

    }
}

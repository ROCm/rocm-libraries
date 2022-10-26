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
            ExpressionPtr m_lhs;

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
                    rv = literal(evaluate(std::make_shared<Expression>(Expr{lhs, rhs})));
                else if(CCommutativeBinary<Expr> && eval_lhs)
                {
                    rv = associativeBinary.call(rhs, evaluate(lhs));
                }
                else if(eval_rhs)
                {
                    rv = associativeBinary.call(lhs, evaluate(rhs));
                }

                if(rv != nullptr)
                    return rv;

                return std::make_shared<Expression>(Expr({lhs, rhs}));
            }

            template <CBinary Expr>
            requires(!CAssociativeBinary<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({call(expr.lhs), call(expr.rhs)}));
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(
                    Expr({call(expr.lhs), call(expr.r1hs), call(expr.r2hs)}));
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({call(expr.arg)}));
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

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        ExpressionPtr fuseAssociative(ExpressionPtr expr)
        {
            auto visitor = AssociativeExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

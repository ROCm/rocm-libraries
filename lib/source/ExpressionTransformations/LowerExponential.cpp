#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Lower precision e^x
         */

        struct LowerExponentialExpressionVisitor
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

            // e^x = exp2(x * log2(e));
            ExpressionPtr operator()(Exponential const& expr) const
            {
                auto arg = call(expr.arg);

                ExpressionPtr log2e    = literal(1.44270f);
                auto          mul_expr = log2e * (expr.arg);
                return std::make_shared<Expression>(Exponential2({mul_expr, expr.comment}));
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
         * Attempts to use lowerExponential for all of the Exponentials within an Expression.
         */
        ExpressionPtr lowerExponential(ExpressionPtr expr)
        {
            auto visitor = LowerExponentialExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct DataFlowTagPropagationVisitor
        {
            DataFlowTagPropagationVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
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

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
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

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(DataFlowTag const& expr)
            {
                if(m_context->registerTagManager()->hasExpression(expr.tag))
                {
                    auto [tagExpr, tagDT]
                        = m_context->registerTagManager()->getExpression(expr.tag);
                    return call(tagExpr);
                }
                else
                {
                    AssertFatal(m_context->registerTagManager()->hasRegister(expr.tag),
                                ShowValue(expr.tag));
                    return std::make_shared<Expression>(
                        m_context->registerTagManager()->getRegister(expr.tag));
                }
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr)
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr)
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            ContextPtr m_context;
        };

        ExpressionPtr dataFlowTagPropagation(ExpressionPtr expr, ContextPtr context)
        {
            auto visitor = DataFlowTagPropagationVisitor(context);
            return visitor.call(expr);
        }

    }
}

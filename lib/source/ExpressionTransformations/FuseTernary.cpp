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

            ExpressionPtr operator()(Add const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                if(eval_lhs && eval_rhs && std::holds_alternative<ShiftL>(*lhs))
                {
                    auto shift   = std::get<ShiftL>(*lhs);
                    auto comment = shift.comment + expr.comment;
                    return std::make_shared<Expression>(
                        ShiftLAdd({shift.lhs, shift.rhs, rhs, comment}));
                }

                if(std::holds_alternative<Multiply>(*lhs))
                {
                    auto multiply = std::get<Multiply>(*lhs);
                    return multiplyAdd(multiply.lhs, multiply.rhs, rhs);
                }

                return std::make_shared<Expression>(Add({lhs, rhs, expr.comment}));
            }

            ExpressionPtr operator()(ShiftL const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                if(eval_lhs && eval_rhs && std::holds_alternative<Add>(*lhs))
                {
                    auto add     = std::get<Add>(*lhs);
                    auto comment = add.comment + expr.comment;
                    return std::make_shared<Expression>(AddShiftL{add.lhs, add.rhs, rhs, comment});
                }
                else
                {
                    return std::make_shared<Expression>(expr);
                }
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

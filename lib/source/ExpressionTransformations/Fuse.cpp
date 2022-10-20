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
         * Fuse binary expressions into available ternary expressions.
         *
         * Fusions:
         * - Add followed by shiftL -> addShiftL
         * - shiftL followed by add -> shiftAddL
         * TODO fused multiply add
         */

        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        struct FuseExpressionVisitor
        {
            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({(*this)(expr.lhs), (*this)(expr.rhs)}));
            }

            // TODO add to binary
            ExpressionPtr operator()(MatrixMultiply const& expr) const
            {
                return std::make_shared<Expression>(MatrixMultiply(expr));
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

                ExpressionPtr rv;
                if(eval_lhs && eval_rhs && std::holds_alternative<ShiftL>(*lhs))
                {
                    auto shift = std::get<ShiftL>(*lhs);
                    return std::make_shared<Expression>(ShiftLAdd({shift.lhs, shift.rhs, rhs}));
                }

                if(rv != nullptr)
                {
                    return rv;
                }

                return std::make_shared<Expression>(Add({lhs, rhs}));
            }

            ExpressionPtr operator()(ShiftL const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                if(eval_lhs && eval_rhs && std::holds_alternative<Add>(*lhs))
                {
                    auto add = std::get<Add>(*lhs);
                    return std::make_shared<Expression>(AddShiftL({add.lhs, add.rhs, rhs}));
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

            ExpressionPtr operator()(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        /**
         * Attempts to use fuse an Expression.
         */
        ExpressionPtr fuse(ExpressionPtr expr)
        {
            auto visitor = FuseExpressionVisitor();
            return visitor(expr);
        }
    }
}

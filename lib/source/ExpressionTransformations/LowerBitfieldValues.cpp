#include <rocRoller/Expression.hpp>
#include <variant>

namespace rocRoller
{
    namespace Expression
    {
        struct LowerBitfieldValuesVisitor
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

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                if constexpr(std::same_as<Value, Register::ValuePtr>)
                {
                    if(expr->isBitfield())
                    {
                        auto info     = DataTypeInfo::Get(expr->variableType());
                        auto dataType = (info.packing > 1) ? info.segmentVariableType.dataType
                                                           : expr->variableType().dataType;
                        return bfe(dataType,
                                   expr->expression(),
                                   expr->getBitOffset(),
                                   expr->getBitWidth());
                    }
                }
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
         *  Replace bitfield ValuePtr expressions with BitFieldExtract expressions
         */
        ExpressionPtr lowerBitfieldValues(ExpressionPtr expr)
        {
            auto visitor = LowerBitfieldValuesVisitor();
            return visitor.call(expr);
        }

    }
}

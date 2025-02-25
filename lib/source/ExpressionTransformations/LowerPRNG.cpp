#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {

        struct LowerPRNGExpressionVisitor
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

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(RandomNumber const& expr) const
            {
                auto arg = call(expr.arg);

                // PRNG algorithm: ((seed << 1) ^ (((seed >> 31) & 1) ? 0xc5 : 0x00))
                ExpressionPtr one = literal(1u);

                ExpressionPtr lhs = std::make_shared<Expression>(ShiftL({arg, one}));

                ExpressionPtr shiftR
                    = std::make_shared<Expression>(LogicalShiftR({arg, literal(31u)}));
                ExpressionPtr bitAnd    = std::make_shared<Expression>(BitwiseAnd({shiftR, one}));
                ExpressionPtr predicate = std::make_shared<Expression>(Equal({bitAnd, one}));

                // Note: here we compute the `xor 0xc5` even though the value might not be selected.
                //       The reason of doing this instead of
                //          rhs = condition(predicate, 0xc5, 0x00)
                //          result = xor(value, rhs)
                //       is that `v_cndmask_b32_e64 v5, 0, 197, s[12:13]`  is not able to assemble
                //       (Error: literal operands are not supported).
                ExpressionPtr xorValue
                    = std::make_shared<Expression>(BitwiseXor({literal(197u), lhs})); // xor 0xc5

                return std::make_shared<Expression>(Conditional({predicate, xorValue, lhs}));
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
         *  Replace RandomNumber expression with equivalent expressions
         */
        ExpressionPtr lowerPRNG(ExpressionPtr expr)
        {
            auto visitor = LowerPRNGExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

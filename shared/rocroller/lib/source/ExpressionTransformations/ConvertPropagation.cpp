/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr applyConvertToValues(Convert const& expr);

        /**
         * Apply conversion to all inputs
         */
        struct ApplyConvertToValuesVisitor
        {
            ApplyConvertToValuesVisitor(DataType datatype)
                : m_destinationType(datatype)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <typename Expr>
            requires CBinary<Expr> &&(!CShift<Expr>)ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            template <CShift Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
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

            ExpressionPtr operator()(Conditional const& expr) const
            {
                auto cpy = expr;
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(AddShiftL const& expr) const
            {
                auto cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ShiftLAdd const& expr) const
            {
                auto cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Convert const& expr) const
            {
                if(expr.destinationType == m_destinationType)
                {
                    auto cpy = expr;
                    cpy.arg  = call(expr.arg);
                    return std::make_shared<Expression>(cpy);
                }
                return applyConvertToValues(expr);
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& value) const
            {
                if constexpr(std::same_as<Value, Register::ValuePtr>)
                {
                    if(value->variableType() == m_destinationType)
                    {
                        return std::make_shared<Expression>(value);
                    }
                    // Only propagate to 64-bit Int
                    if(value->variableType() == DataType::Int64
                       || value->variableType() == DataType::UInt64)
                    {
                        return convert(m_destinationType, std::make_shared<Expression>(value));
                    }
                }
                return std::make_shared<Expression>(value);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            DataType m_destinationType;
        };

        /**
         * Traverses the tree and applies convert propagation to the first valid convert node
         * along every root-to-leaf path.
         */
        struct ConvertPropagationVisitor
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

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
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

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(Convert const& expr) const
            {
                return applyConvertToValues(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                AssertFatal(expr != nullptr, "Found nullptr in expression");
                return std::visit(*this, *expr);
            }
        };

        ExpressionPtr applyConvertToValues(Convert const& expr)
        {
            if(expr.destinationType == DataType::UInt32 || expr.destinationType == DataType::Int32)
            {
                auto visitor = ApplyConvertToValuesVisitor(expr.destinationType);
                return visitor.call(std::make_shared<Expression>(expr));
            }
            return std::make_shared<Expression>(expr);
        }

        /**
         * Propagate converts to inputs
         * e.g. Given A and B are Int64,
         * Int32(A + B) gets converted to Int32(Int32(A) + Int32(B))
         * to avoid unnecessary 64-bit arithmetic
         */
        ExpressionPtr convertPropagation(ExpressionPtr expr)
        {
            auto visitor = ConvertPropagationVisitor();
            return visitor.call(expr);
        }

        struct MakeScalarVisitor
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

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
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

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(Register::ValuePtr const& expr) const
            {
                if(expr->regType() == Register::Type::Vector)
                    return std::make_shared<Expression>(ToScalar{expr->expression()});

                return expr->expression();
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                AssertFatal(expr != nullptr, "Found nullptr in expression");
                return std::visit(*this, *expr);
            }
        };

        ExpressionPtr makeScalar(ExpressionPtr expr)
        {
            MakeScalarVisitor visitor;
            return visitor.call(expr);
        }
    }
}

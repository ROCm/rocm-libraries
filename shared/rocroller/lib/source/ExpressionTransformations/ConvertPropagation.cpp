// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
                AssertFatal(
                    m_destinationType == DataType::UInt32 or m_destinationType == DataType::Int32,
                    "ApplyConvertToValuesVisitor only allows destinationType to be UInt32 or Int32",
                    ShowValue(m_destinationType));
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
                if constexpr(std::same_as<Expr, Divide> or std::same_as<Expr, Modulo>)
                {
                    auto const resultDataType = resultVariableType(expr).dataType;
                    if(DataTypeInfo::Get(resultDataType).elementBytes
                       > DataTypeInfo::Get(m_destinationType).elementBytes)
                    {
                        auto const typeInfo = DataTypeInfo::Get(resultDataType);
                        AssertFatal(typeInfo.isIntegral,
                                    "Converting a non-integral to integral might result in loss of "
                                    "precision");
                        return convert(m_destinationType, std::make_shared<Expression>(expr));
                    }
                }

                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            template <CShift Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                if constexpr(CIsAnyOf<Expr, LogicalShiftR, ArithmeticShiftR>)
                {
                    auto const resultDataType = resultVariableType(expr).dataType;
                    if(DataTypeInfo::Get(resultDataType).elementBytes
                       > DataTypeInfo::Get(m_destinationType).elementBytes)
                    {
                        return convert(m_destinationType, std::make_shared<Expression>(expr));
                    }
                }

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

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
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

                if((m_destinationType == DataType::UInt32 or m_destinationType == DataType::Int32)
                   and (expr.destinationType == DataType::UInt64
                        or expr.destinationType == DataType::Int64))
                {
                    auto cpy            = expr;
                    cpy.destinationType = m_destinationType;
                    cpy.arg             = call(expr.arg);
                    return std::make_shared<Expression>(cpy);
                }

                return applyConvertToValues(expr);
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& value) const
            {
                // Operating on CommandArgumentPtr causes more scalar loads at start of kernel
                // leading to running out of SGPRs
                if constexpr(not std::same_as<Value, CommandArgumentPtr>)
                {
                    const auto variableType
                        = resultVariableType(std::make_shared<Expression>(value));
                    if(variableType != m_destinationType
                       && (variableType == DataType::Int64 || variableType == DataType::UInt64))
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

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
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

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
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

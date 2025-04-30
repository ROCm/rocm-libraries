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

#include <common/WidenAddrExprTo64bit.hpp>
#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/Expression.hpp>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRollerTest
{

    using namespace rocRoller;

    struct WidenTo64BitVisitor
    {

        Expression::ExpressionPtr operator()(Expression::Convert const& expr) const
        {
            Expression::Convert cpy = expr;
            if(expr.arg)
            {
                // Here is an assumption that call(cpy.arg) never goes above 64-bit and
                // input convert's destination types are either int32, uint32, int64 or uint64.
                // resultVariableType(expr.arg) is not called intentionally as it will visit the
                // subtree of expr.arg again.
                cpy.arg = call(expr.arg);
                if(expr.destinationType == DataType::UInt32)
                    return convert(DataType::UInt64, cpy.arg);
                else if(expr.destinationType == DataType::Int32)
                    return convert(DataType::Int64, cpy.arg);
                else if(expr.destinationType == DataType::UInt64
                        || expr.destinationType == DataType::Int64)
                    return convert(expr.destinationType, cpy.arg);
                else
                {
                    AssertFatal(false,
                                "Expected Destination type for a convert is either (u)int{32,64}");
                    return nullptr;
                }
            }

            return std::make_shared<Expression::Expression>(cpy);
        }

        Expression::ExpressionPtr operator()(Expression::Negate const& expr) const
        {
            Expression::Negate cpy = expr;
            if(expr.arg)
            {
                cpy.arg = call(expr.arg);
            }

            return std::make_shared<Expression::Expression>(cpy);
        }

        template <Expression::CUnary Expr>
        Expression::ExpressionPtr operator()(Expr const& expr) const
        {
            // Only Expression::Convert and perhaps negates are expected.
            AssertFatal(false, "Unexpected Unary expression", ShowValue(expr));
            return nullptr;
        }

        Expression::ExpressionPtr operator()(Expression::Divide const& expr) const
        {
            // For 1. Make sure divisor is a CValue other than DataFlowTag and WaveTilePtr.
            //     2. The dataType is not 64-bit. But this check is done by fast math's
            //        power of 2 division/modulo anyway.
            //        So no need to do it here.
            // Note that the rhs operand is not extended to 64-bit as fast division/modulo
            // of 64bit divisor is not implemented.
            // But we make sure the divisor is within 32-bit literal or register value.
            Log::debug("Divisor: {} ", toString(expr.rhs));
            AssertFatal(isExpectedLeafType(expr.rhs), "Divisor should be a leaf value.");

            Expression::Divide cpy = expr;
            if(expr.lhs)
                cpy.lhs = call(expr.lhs);

            return std::make_shared<Expression::Expression>(cpy);
        }

        Expression::ExpressionPtr operator()(Expression::Modulo const& expr) const
        {
            // For 1. Make sure divisor is a CValue other than DataFlowTag and WaveTilePtr.
            //     2. The dataType is not 64-bit. But this check is done by fast math's
            //        power of 2 division/modulo anyway.
            //        So no need to do it here.
            // Note that the rhs operand is not extended to 64-bit as fast division/modulo
            // of 64bit divisor is not implemented.
            // But we make sure the divisor is within 32-bit literal or register value.
            Log::debug("Modulo: {} ", toString(expr.rhs));
            AssertFatal(isExpectedLeafType(expr.rhs),
                        "Second operand of Modulo should be a leaf value");

            Expression::Modulo cpy = expr;
            if(expr.lhs)
                cpy.lhs = call(expr.lhs);

            return std::make_shared<Expression::Expression>(cpy);
        }

        // TODO: Shift operations might needed to be avoided as well.
        //       Address calculation expressions before fastMath applied
        //       probably don't contain shift operations either.
        template <Expression::CBinary Expr>
        requires(Expression::CArithmetic<Expr>) Expression::ExpressionPtr
            operator()(Expr const& expr) const
        {
            if constexpr(std::same_as<Expr, Expression::Subtract>)
            {
                AssertFatal(false, "Subtracts are not expected");
                return nullptr;
            }

            if constexpr(Expression::CLogical<Expr>)
            {
                AssertFatal(false, "logicals are not expected");
                return nullptr;
            }

            Expr cpy = expr;
            if(expr.lhs)
                cpy.lhs = call(expr.lhs);
            if(expr.rhs)
                cpy.rhs = call(expr.rhs);

            return std::make_shared<Expression::Expression>(cpy);
        }

        // Even with catch-all operator()(Expression::ExpressionPtr) without following,
        // compilation fails.
        template <typename Expr>
        requires(Expression::CBinary<Expr>) Expression::ExpressionPtr
            operator()(Expr const& expr) const
        {
            AssertFatal(false, "Not expected expr : ", ShowValue(expr));
            return nullptr;
        }

        template <Expression::CTernary Expr>
        requires(Expression::CArithmetic<Expr>) Expression::ExpressionPtr
            operator()(Expr const& expr) const
        {
            Expr cpy = expr;
            if(expr.lhs)
                cpy.lhs = call(expr.lhs);
            if(expr.r1hs)
                cpy.r1hs = call(expr.r1hs);
            if(expr.r2hs)
                cpy.r2hs = call(expr.r2hs);

            return std::make_shared<Expression::Expression>(cpy);
        }

        template <typename Expr>
        requires(Expression::CTernary<Expr>) Expression::ExpressionPtr
            operator()(Expr const& expr) const
        {
            AssertFatal(false, "Not expected expr : ", ShowValue(expr));
            return nullptr;
        }

        // leaves
        Expression::ExpressionPtr operator()(CommandArgumentPtr const& expr) const
        {
            Log::debug("CommandArgumentPtr {}", Expression::toString(expr));

            auto varType = expr->variableType();

            assertIfNotExpectedType(varType.dataType, Expression::toString(expr));

            CommandArgumentPtr cpy = expr;
            return widenTo64(varType.dataType, cpy);
        }

        Expression::ExpressionPtr operator()(CommandArgumentValue const& expr) const
        {
            Log::debug("CommandArgumentValue {} Type {} ",
                       Expression::toString(expr),
                       toString(variableType(expr)));

            auto varType = variableType(expr);

            assertIfNotExpectedType(varType.dataType, Expression::toString(expr));

            CommandArgumentValue cpy = expr;
            return widenTo64(varType.dataType, cpy);
        }

        Expression::ExpressionPtr operator()(Register::ValuePtr const& expr) const
        {
            Log::debug("Register::ValuePtr {}", Expression::toString(expr));

            auto varType = expr->variableType();

            assertIfNotExpectedType(varType.dataType, Expression::toString(expr));

            Register::ValuePtr cpy = expr;
            return widenTo64(varType.dataType, cpy);
        }

        Expression::ExpressionPtr operator()(AssemblyKernelArgumentPtr const& expr) const
        {
            Log::debug("AssemblyKernelArgumentPtr {} its expression is {}",
                       Expression::toString(expr),
                       Expression::toString(expr->expression));

            auto varType = expr->variableType;

            assertIfNotExpectedType(varType.dataType, Expression::toString(expr));

            AssemblyKernelArgumentPtr cpy = expr;
            return widenTo64(varType.dataType, cpy);
        }

        // catch the rest CValue
        template <Expression::CValue Value>
        Expression::ExpressionPtr operator()(Value const& expr) const
        {
            AssertFatal(
                false, "No expectation to meet WaveTilePtr or DataFlowTag : ", ShowValue(expr));
            return nullptr;
        }

        Expression::ExpressionPtr operator()(Expression::Expression const& expr) const
        {
            AssertFatal(false, "No expectation to meet this type of Expression: ", ShowValue(expr));
            return nullptr;
        }

        template <typename T>
        Expression::ExpressionPtr widenTo64(DataType srcType, T const& expr) const
        {
            if(srcType == DataType::UInt32)
                return convert(DataType::UInt64, std::make_shared<Expression::Expression>(expr));
            else if(srcType == DataType::Int32)
                return convert(DataType::Int64, std::make_shared<Expression::Expression>(expr));

            return std::make_shared<Expression::Expression>(expr);
        }

        Expression::ExpressionPtr call(Expression::ExpressionPtr const& expr) const
        {
            return std::visit(*this, *expr);
        }

        void assertIfNotExpectedType(DataType dt, std::string const& showValue) const
        {
            AssertFatal(dt == DataType::Int32 || dt == DataType::UInt32 || dt == DataType::UInt64
                            || dt == DataType::Int64,
                        "Unexpected DataType for Command/Kernel arguments or "
                        "workgroup/item indices ",
                        showValue);
        }

        bool isExpectedLeafType(Expression::ExpressionPtr const& expr) const
        {
            return std::holds_alternative<CommandArgumentValue>(*expr)
                   || std::holds_alternative<CommandArgumentPtr>(*expr)
                   || std::holds_alternative<Register::ValuePtr>(*expr)
                   || std::holds_alternative<AssemblyKernelArgumentPtr>(*expr);
        }
    };

    Expression::ExpressionPtr widenAddrExprTo64bit(Expression::ExpressionPtr expr)
    {
        auto origVarType = resultVariableType(expr);

        auto visitor = WidenTo64BitVisitor();
        auto widened = visitor.call(expr);

        auto finalVarType = resultVariableType(widened);

        AssertFatal(origVarType.dataType == finalVarType.dataType,
                    "Original and final data types should be the same",
                    ShowValue(origVarType.dataType),
                    ShowValue(finalVarType.dataType),
                    ShowValue(expr),
                    ShowValue(widened));

        return widened;
    }
}

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2024 Advanced Micro Devices, Inc.
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

#pragma once

#include <memory>
#include <stack>
#include <variant>

#include "CodeGen/Instruction.hpp"
#include "InstructionValues/Register.hpp"
#include "Operations/CommandArgument.hpp"
#include "Utilities/Component.hpp"

namespace rocRoller
{

    namespace Expression
    {
        inline ExpressionPtr operator+(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Add{a, b});
        }

        inline ExpressionPtr operator-(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Subtract{a, b});
        }

        inline ExpressionPtr operator*(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Multiply{a, b});
        }

        inline ExpressionPtr operator/(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Divide{a, b});
        }

        inline ExpressionPtr operator%(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Modulo{a, b});
        }

        inline ExpressionPtr operator<<(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(ShiftL{a, b});
        }

        inline ExpressionPtr operator>>(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(ArithmeticShiftR{a, b});
        }

        inline ExpressionPtr arithmeticShiftR(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(ArithmeticShiftR{a, b});
        }

        inline ExpressionPtr logicalShiftR(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LogicalShiftR{a, b});
        }

        inline ExpressionPtr operator&(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseAnd{a, b});
        }

        inline ExpressionPtr operator|(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseOr{a, b});
        }

        inline ExpressionPtr operator^(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseXor{a, b});
        }

        inline ExpressionPtr operator>(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(GreaterThan{a, b});
        }

        inline ExpressionPtr operator>=(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(GreaterThanEqual{a, b});
        }

        inline ExpressionPtr operator<(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LessThan{a, b});
        }

        inline ExpressionPtr operator<=(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LessThanEqual{a, b});
        }

        inline ExpressionPtr operator==(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Equal{a, b});
        }

        inline ExpressionPtr operator!=(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(NotEqual{a, b});
        }

        inline ExpressionPtr operator&&(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LogicalAnd{a, b});
        }

        inline ExpressionPtr operator||(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LogicalOr{a, b});
        }

        inline ExpressionPtr logicalNot(ExpressionPtr a)
        {
            return std::make_shared<Expression>(LogicalNot{a});
        }

        inline ExpressionPtr operator-(ExpressionPtr a)
        {
            return std::make_shared<Expression>(Negate{a});
        }

        inline ExpressionPtr operator~(ExpressionPtr a)
        {
            return std::make_shared<Expression>(BitwiseNegate{a});
        }

        inline ExpressionPtr exp2(ExpressionPtr a)
        {
            return std::make_shared<Expression>(Exponential2{a});
        }

        inline ExpressionPtr exp(ExpressionPtr a)
        {
            return std::make_shared<Expression>(Exponential{a});
        }

        inline ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(MultiplyHigh{a, b});
        }

        inline ExpressionPtr multiplyAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(MultiplyAdd{a, b, c});
        }

        inline ExpressionPtr addShiftL(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(AddShiftL{a, b, c});
        }

        inline ExpressionPtr shiftLAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(ShiftLAdd{a, b, c});
        }

        inline ExpressionPtr conditional(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(Conditional{a, b, c});
        }

        inline ExpressionPtr magicMultiple(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicMultiple{a});
        }

        inline ExpressionPtr magicShifts(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicShifts{a});
        }

        inline ExpressionPtr magicSign(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicSign{a});
        }

        template <int idx = 0>
        inline ExpressionPtr convertCase(DataType dt, ExpressionPtr a)
        {
            // Template-based recursion "loop" to automatically convert runtime
            // enumeration to template parameter.  Depends on DataType having a
            // Count enumeration which will terminate the recursion.

            // This makes it so that if a value is added to the DataType enumeration,
            // it will automatically be checked here.

            constexpr auto theDT = static_cast<DataType>(idx);

            if(dt == theDT)
                return convert<theDT>(a);

            if constexpr(idx + 1 < static_cast<int>(DataType::Count))
                return convertCase<idx + 1>(dt, a);

            Throw<FatalError>("Unsupported datatype conversion: ", ShowValue(dt));
        }

        inline ExpressionPtr convert(DataType dt, ExpressionPtr a)
        {
            return convertCase(dt, a);
        }

        template <DataType DATATYPE>
        inline ExpressionPtr convert(ExpressionPtr a)
        {
            // Check if Convert<DATATYPE> is an alternative of the Expression variant type.
            // This means this function doesn't need to change if we add a new Convert<>
            // alternative to the Expression variant type.
            if constexpr(CExpression<Convert<DATATYPE>>)
                return std::make_shared<Expression>(Convert<DATATYPE>{a});

            Throw<FatalError>("Conversions to ", DATATYPE, " not yet supported.");
        }

        template <CCommandArgumentValue T>
        inline ExpressionPtr literal(T value)
        {
            return std::make_shared<Expression>(value);
        }

        template <CCommandArgumentValue T>
        ExpressionPtr literal(T value, VariableType v)
        {
            AssertFatal(v.pointerType == PointerType::Value);

            switch(v.dataType)
            {
            case DataType::Int32:
                return literal<int32_t>(value);
            case DataType::UInt32:
                return literal<uint32_t>(value);
            case DataType::Int64:
                return literal<int64_t>(value);
            case DataType::UInt64:
                return literal<uint64_t>(value);
            case DataType::Bool:
                return literal<bool>(value);
            case DataType::Half:
                return literal<Half>(static_cast<float>(value));
            case DataType::Float:
                return literal<float>(value);
            case DataType::Double:
                return literal<double>(value);
            default:
                Throw<FatalError>(
                    "Unsupported datatype ", v.dataType, " provided to Expression::literal");
            }
        }

        static_assert(CExpression<Add>);
        static_assert(!CExpression<Register::Value>,
                      "ValuePtr can be an Expression but Value cannot.");

        template <typename Expr>
        struct ExpressionInfo
        {
        };

#define EXPRESSION_INFO_CUSTOM(cls, cls_name) \
    template <>                               \
    struct ExpressionInfo<cls>                \
    {                                         \
        constexpr static auto name()          \
        {                                     \
            return cls_name;                  \
        }                                     \
    }

#define EXPRESSION_INFO(cls) EXPRESSION_INFO_CUSTOM(cls, #cls)

        EXPRESSION_INFO(Add);
        EXPRESSION_INFO(Subtract);
        EXPRESSION_INFO(MatrixMultiply);
        EXPRESSION_INFO(Multiply);
        EXPRESSION_INFO(MultiplyAdd);
        EXPRESSION_INFO(MultiplyHigh);

        EXPRESSION_INFO(Divide);
        EXPRESSION_INFO(Modulo);

        EXPRESSION_INFO(ShiftL);
        EXPRESSION_INFO(LogicalShiftR);
        EXPRESSION_INFO(ArithmeticShiftR);

        EXPRESSION_INFO(BitwiseNegate);
        EXPRESSION_INFO(BitwiseAnd);
        EXPRESSION_INFO(BitwiseOr);
        EXPRESSION_INFO(BitwiseXor);
        EXPRESSION_INFO(Exponential2);
        EXPRESSION_INFO(Exponential);

        EXPRESSION_INFO(ShiftLAdd);
        EXPRESSION_INFO(AddShiftL);

        EXPRESSION_INFO(Conditional);

        EXPRESSION_INFO(GreaterThan);
        EXPRESSION_INFO(GreaterThanEqual);
        EXPRESSION_INFO(LessThan);
        EXPRESSION_INFO(LessThanEqual);
        EXPRESSION_INFO(Equal);
        EXPRESSION_INFO(NotEqual);
        EXPRESSION_INFO(LogicalAnd);
        EXPRESSION_INFO(LogicalOr);
        EXPRESSION_INFO(LogicalNot);

        EXPRESSION_INFO(MagicMultiple);
        EXPRESSION_INFO(MagicShifts);
        EXPRESSION_INFO(MagicSign);

        EXPRESSION_INFO(Negate);

        EXPRESSION_INFO(Convert<DataType::Half>);
        EXPRESSION_INFO(Convert<DataType::Halfx2>);
        EXPRESSION_INFO(Convert<DataType::FP8x4>);
        EXPRESSION_INFO(Convert<DataType::BF8x4>);
        EXPRESSION_INFO(Convert<DataType::Float>);
        EXPRESSION_INFO(Convert<DataType::Double>);
        EXPRESSION_INFO(Convert<DataType::Int32>);
        EXPRESSION_INFO(Convert<DataType::Int64>);
        EXPRESSION_INFO(Convert<DataType::UInt32>);
        EXPRESSION_INFO(Convert<DataType::UInt64>);

        EXPRESSION_INFO_CUSTOM(Register::ValuePtr, "Register Value");
        EXPRESSION_INFO_CUSTOM(CommandArgumentPtr, "Command Argument");
        EXPRESSION_INFO_CUSTOM(CommandArgumentValue, "Literal Value");
        EXPRESSION_INFO_CUSTOM(AssemblyKernelArgumentPtr, "Kernel Argument");
        EXPRESSION_INFO_CUSTOM(WaveTilePtr, "WaveTile");

        EXPRESSION_INFO(DataFlowTag);

#undef EXPRESSION_INFO
#undef EXPRESSION_INFO_CUSTOM
        struct ExpressionNameVisitor
        {
            template <CExpression Expr>
            std::string operator()(Expr const& expr) const
            {
                return ExpressionInfo<Expr>::name();
            }

            std::string operator()(CommandArgumentPtr const& expr) const
            {
                if(expr)
                    return expr->name();

                return ExpressionInfo<CommandArgumentPtr>::name();
            }

            std::string call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            std::string call(ExpressionPtr const& expr) const
            {
                return call(*expr);
            }
        };

        inline std::string name(ExpressionPtr const& expr)
        {
            return ExpressionNameVisitor().call(expr);
        }

        inline std::string name(Expression const& expr)
        {
            return ExpressionNameVisitor().call(expr);
        }

        struct ExpressionEvaluationTimesVisitor
        {
            EvaluationTimes operator()(WaveTilePtr const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            template <CTernary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                auto lhs  = call(expr.lhs);
                auto r1hs = call(expr.r1hs);
                auto r2hs = call(expr.r2hs);

                return lhs & r1hs & r2hs & Expr::EvalTimes;
            }

            template <CBinary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                return lhs & rhs & Expr::EvalTimes;
            }

            template <CUnary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                return call(expr.arg) & Expr::EvalTimes;
            }

            EvaluationTimes operator()(Register::ValuePtr const& expr) const
            {
                if(expr->regType() == Register::Type::Literal)
                    return EvaluationTimes::All();

                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                return {EvaluationTime::KernelLaunch, EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(DataFlowTag const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(CommandArgumentPtr const& expr) const
            {
                return {EvaluationTime::KernelLaunch};
            }

            EvaluationTimes operator()(CommandArgumentValue const& expr) const
            {
                return EvaluationTimes::All();
            }

            EvaluationTimes call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            EvaluationTimes call(ExpressionPtr const& expr) const
            {
                if(expr == nullptr)
                {
                    return EvaluationTimes::All();
                }
                return call(*expr);
            }
        };

        inline EvaluationTimes evaluationTimes(Expression const& expr)
        {
            return ExpressionEvaluationTimesVisitor().call(expr);
        }

        inline EvaluationTimes evaluationTimes(ExpressionPtr const& expr)
        {
            return ExpressionEvaluationTimesVisitor().call(expr);
        }
    }
}

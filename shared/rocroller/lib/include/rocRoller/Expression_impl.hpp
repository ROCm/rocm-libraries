/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{

    namespace Expression
    {
        inline std::string toString(EvaluationTime t)
        {
            switch(t)
            {
            case EvaluationTime::Translate:
                return "Translate";
            case EvaluationTime::KernelLaunch:
                return "KernelLaunch";
            case EvaluationTime::KernelExecute:
                return "KernelExecute";
            default:
                break;
            }
            Throw<FatalError>("Invalid EvaluationTime");
        }

        inline std::ostream& operator<<(std::ostream& stream, EvaluationTime const& t)
        {
            return stream << toString(t);
        }

        inline std::string toString(AlgebraicProperty p)
        {
            switch(p)
            {
            case AlgebraicProperty::Commutative:
                return "Commutative";
            case AlgebraicProperty::Associative:
                return "Associative";
            default:
                break;
            }
            Throw<FatalError>("Invalid AlgebraicProperty");
        }

        inline std::ostream& operator<<(std::ostream& stream, AlgebraicProperty const& p)
        {
            return stream << toString(p);
        }

        inline std::string toString(Category c)
        {
            switch(c)
            {
            case Category::Arithmetic:
                return "Arithmetic";
            case Category::Comparison:
                return "Comparison";
            case Category::Logical:
                return "Logical";
            case Category::Conversion:
                return "Conversion";
            case Category::Value:
                return "Value";
            default:
                break;
            }
            Throw<FatalError>("Invalid Category");
        }

        inline std::ostream& operator<<(std::ostream& stream, Category const& c)
        {
            return stream << toString(c);
        }

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

        inline ExpressionPtr magicShiftAndSign(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicShiftAndSign{a});
        }

        inline static bool convertibleTo(DataType dt)
        {
            return dt == DataType::Half || dt == DataType::Halfx2 || dt == DataType::BFloat16
                   || dt == DataType::BFloat16x2 || dt == DataType::FP8 || dt == DataType::BF8
                   || dt == DataType::FP8x4 || dt == DataType::BF8x4 || dt == DataType::Float
                   || dt == DataType::FP6x16 || dt == DataType::BF6x16 || dt == DataType::FP4x8
                   || dt == DataType::Double || dt == DataType::Int32 || dt == DataType::Int64
                   || dt == DataType::UInt32 || dt == DataType::UInt64 || dt == DataType::Bool
                   || dt == DataType::Bool32 || dt == DataType::Bool64;
        }

        inline ExpressionPtr convert(DataType dt, ExpressionPtr a)
        {
            // Couldn't use previous impl. of CExpression<T>
            // (aka. std::constructible_from<Expression, T>) because Convert<DATATYPE>
            // is not a type anymore. Convert is no longer templated.
            // Either a runtime-check or other templatized function call is needed
            // to make sure valid destination types for a convert.
            // Currently, explicit runtime check is used. (Notice that checking the range
            // of dt doesn't work because target type of convert is not consecutively
            // laid in DataType enum.)
            if(!convertibleTo(dt))
                Throw<FatalError>("Expression - Unsupported datatype conversion: ", ShowValue(dt));

            return std::make_shared<Expression>(Convert{{.arg{a}}, dt});
        }

        inline ExpressionPtr convert(VariableType vt, ExpressionPtr a)
        {
            AssertFatal(!vt.isPointer(), "Convert to pointer type not supported.", ShowValue(vt));

            return convert(vt.dataType, a);
        }

        template <DataType DATATYPE>
        inline ExpressionPtr convert(ExpressionPtr a)
        {
            return convert(DATATYPE, a);
        }

        inline ExpressionPtr bfe(ExpressionPtr a, uint8_t offset, uint8_t width)
        {
            return std::make_shared<Expression>(
                BitFieldExtract{{.arg{a}}, DataType::None, offset, width});
        }

        inline ExpressionPtr bfe(DataType dt, ExpressionPtr a, uint8_t offset, uint8_t width)
        {
            return std::make_shared<Expression>(BitFieldExtract{{.arg{a}}, dt, offset, width});
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
        EXPRESSION_INFO(ScaledMatrixMultiply);
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
        EXPRESSION_INFO(MagicShiftAndSign);

        EXPRESSION_INFO(Negate);

        EXPRESSION_INFO(RandomNumber);

        EXPRESSION_INFO(ToScalar);

        EXPRESSION_INFO(BitFieldExtract);

        EXPRESSION_INFO(Convert);

        EXPRESSION_INFO_CUSTOM(SRConvert<DataType::FP8>, "SRConvert_FP8");
        EXPRESSION_INFO_CUSTOM(SRConvert<DataType::BF8>, "SRConvert_BF8");

        EXPRESSION_INFO_CUSTOM(Register::ValuePtr, "RegisterValue");
        EXPRESSION_INFO_CUSTOM(CommandArgumentPtr, "CommandArgument");
        EXPRESSION_INFO_CUSTOM(CommandArgumentValue, "LiteralValue");
        EXPRESSION_INFO_CUSTOM(AssemblyKernelArgumentPtr, "Kernel Argument");
        EXPRESSION_INFO_CUSTOM(WaveTilePtr, "WaveTile");

        EXPRESSION_INFO(DataFlowTag);
        EXPRESSION_INFO(PositionalArgument);

#undef EXPRESSION_INFO
#undef EXPRESSION_INFO_CUSTOM
        struct ExpressionNameVisitor
        {
            template <CExpression Expr>
            std::string operator()(Expr const& expr) const
            {
                return ExpressionInfo<Expr>::name();
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

        struct ExpressionArgumentNameVisitor
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

        inline std::string argumentName(ExpressionPtr const& expr)
        {
            return ExpressionArgumentNameVisitor().call(expr);
        }

        inline std::string argumentName(Expression const& expr)
        {
            return ExpressionArgumentNameVisitor().call(expr);
        }

        struct ExpressionEvaluationTimesVisitor
        {
            EvaluationTimes operator()(WaveTilePtr const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(ScaledMatrixMultiply const& expr) const
            {
                auto matA   = call(expr.matA);
                auto matB   = call(expr.matB);
                auto matC   = call(expr.matC);
                auto scaleA = call(expr.scaleA);
                auto scaleB = call(expr.scaleB);

                return matA & matB & matC & scaleA & scaleB & ScaledMatrixMultiply::EvalTimes;
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

            constexpr EvaluationTimes operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                return {EvaluationTime::KernelLaunch, EvaluationTime::KernelExecute};
            }

            constexpr EvaluationTimes operator()(DataFlowTag const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            constexpr EvaluationTimes operator()(PositionalArgument const& expr) const
            {
                return {EvaluationTime::Translate};
            }

            constexpr EvaluationTimes operator()(CommandArgumentPtr const& expr) const
            {
                return {EvaluationTime::KernelLaunch};
            }

            constexpr EvaluationTimes operator()(CommandArgumentValue const& expr) const
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

        template <typename Expr>
        requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) auto split(ExpressionPtr expr)
        {
            AssertFatal(expr && std::holds_alternative<Expr>(*expr),
                        "Expression does not hold the correct type");

            auto exp = std::get<Expr>(*expr);

            if constexpr(CUnary<Expr>)
            {
                return std::make_tuple(exp.arg);
            }
            else if constexpr(CBinary<Expr>)
            {
                return std::make_tuple(exp.lhs, exp.rhs);
            }
            else if constexpr(CTernary<Expr>)
            {
                return std::make_tuple(exp.lhs, exp.r1hs, exp.r2hs);
            }
        }

    }
}

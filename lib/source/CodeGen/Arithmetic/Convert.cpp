/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/Arithmetic/Convert.hpp>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponent(ConvertGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert>>
        GetGenerator<Expression::Convert>(Register::ValuePtr dst,
                                          Register::ValuePtr arg,
                                          Expression::Convert const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Convert>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction>
        generateConvertOp(DataType dataType, Register::ValuePtr dest, Register::ValuePtr arg)
    {
        // The third Convert Expression object is used only for passing result DataType information.
        co_yield generateOp<Expression::Convert>(
            dest, arg, Expression::Convert{.destinationType = dataType});
    }

    Generator<Instruction> ConvertGenerator::generate(Register::ValuePtr         dest,
                                                      Register::ValuePtr         arg,
                                                      Expression::Convert const& expr)
    {
#define ConvertCase(dtype)                   \
    case DataType::dtype:                    \
        co_yield generate##dtype(dest, arg); \
        break

        switch(expr.destinationType)
        {
            ConvertCase(Float);
            ConvertCase(Half);
            ConvertCase(Halfx2);
            ConvertCase(BFloat16);
            ConvertCase(BFloat16x2);
            ConvertCase(FP8);
            ConvertCase(BF8);
            ConvertCase(Int32);
            ConvertCase(Int64);
            ConvertCase(UInt32);
            ConvertCase(UInt64);
            ConvertCase(FP8x4);
            ConvertCase(BF8x4);
            ConvertCase(FP6x16);
            ConvertCase(BF6x16);
            ConvertCase(FP4x8);
            ConvertCase(Double);

        default:
            Throw<FatalError>("Generate - Unsupported datatype conversion: ",
                              ShowValue(expr.destinationType));
        }
#undef ConvertCase
    }

    Generator<Instruction> ConvertGenerator::generateFloat(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Half:
            co_yield_(Instruction("v_cvt_f32_f16", {dest}, {arg}, {}, ""));
            break;
        case DataType::Halfx2:
            co_yield_(Instruction("v_cvt_f32_f16", {dest->element({0})}, {arg}, {}, ""));
            co_yield generateOp<Expression::LogicalShiftR>(
                dest->element({1}), arg, Register::Value::Literal(16u));
            co_yield_(
                Instruction("v_cvt_f32_f16", {dest->element({1})}, {dest->element({1})}, {}, ""));
            break;
        case DataType::FP8:
            co_yield_(Instruction("v_cvt_f32_fp8", {dest}, {arg}, {}, ""));
            break;
        case DataType::BF8:
            co_yield_(Instruction("v_cvt_f32_bf8", {dest}, {arg}, {}, ""));
            break;
        case DataType::BFloat16:
            co_yield generateOp<Expression::ShiftL>(dest, arg, Register::Value::Literal(16u));
            break;
        case DataType::BFloat16x2:
            // unpack BFloat16x2
            co_yield generateOp<Expression::BitwiseAnd>(
                dest->element({0}), arg, Register::Value::Literal(0xFFFF));
            co_yield generateOp<Expression::LogicalShiftR>(
                dest->element({1}), arg, Register::Value::Literal(16u));
            // convert BFloat16 to FP32
            co_yield generateOp<Expression::ShiftL>(
                dest->element({0}), dest->element({0}), Register::Value::Literal(16u));
            co_yield generateOp<Expression::ShiftL>(
                dest->element({1}), dest->element({1}), Register::Value::Literal(16u));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to float: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateHalf(Register::ValuePtr dest,
                                                          Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        // conversion cannot operate on ACCVGPR
        AssertFatal(dest->regType() != rocRoller::Register::Type::Accumulator);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
            if(arg->regType() == rocRoller::Register::Type::Accumulator)
            {
                const auto& arch = m_context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasAccCD),
                            concatenate("Architecture",
                                        arch.target().toString(),
                                        "does not use Accumulator registers."));
                // If arg is ACCVGPR, we first copy the value to dest (Vector)
                // and then convert.
                co_yield m_context->copier()->copy(dest, arg, "");
                co_yield_(Instruction("v_cvt_f16_f32", {dest}, {dest}, {}, ""));
            }
            else
            {
                co_yield_(Instruction("v_cvt_f16_f32", {dest}, {arg}, {}, ""));
            }
            break;
        case DataType::Halfx2:
            co_yield generateOp<Expression::BitwiseAnd>(
                dest->element({0}), arg, Register::Value::Literal(0xFFFF));
            co_yield generateOp<Expression::LogicalShiftR>(
                dest->element({1}), arg, Register::Value::Literal(16u));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to half: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateHalfx2(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
        {
            AssertFatal(arg->valueCount() == 2,
                        "Conversion to Halfx2 requires two elements (",
                        arg->description(),
                        ")");

            auto temp = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            co_yield generateHalf(dest, arg->element({0}));
            co_yield generateHalf(temp, arg->element({1}));

            co_yield m_context->copier()->packHalf(dest, dest, temp);
            break;
        }
        case DataType::Half:
            AssertFatal(arg->valueCount() == 2,
                        "Conversion to Halfx2 requires two elements (",
                        arg->description(),
                        ")");
            co_yield m_context->copier()->packHalf(dest, arg->element({0}), arg->element({1}));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to halfx2: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateBFloat16(Register::ValuePtr dest,
                                                              Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
            co_yield generateOp<Expression::LogicalShiftR>(
                dest, arg, Register::Value::Literal(16u));
            break;
        case DataType::BFloat16x2:
            co_yield generateOp<Expression::BitwiseAnd>(
                dest->element({0}), arg, Register::Value::Literal(0xFFFF));
            co_yield generateOp<Expression::LogicalShiftR>(
                dest->element({1}), arg, Register::Value::Literal(16u));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to bfloat16: ",
                              ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateBFloat16x2(Register::ValuePtr dest,
                                                                Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
        {
            AssertFatal(arg->valueCount() == 2,
                        "Conversion to Bfloat16x2 requires two elements (",
                        arg->description(),
                        ")");

            auto temp = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::BFloat16, 1);

            co_yield generateBFloat16(dest, arg->element({0}));
            co_yield generateBFloat16(temp, arg->element({1}));

            co_yield m_context->copier()->packHalf(dest, dest, temp);
            break;
        }
        case DataType::BFloat16:
            AssertFatal(arg->valueCount() == 2, "Conversion to Bfloat16x2 requires two elements");
            co_yield m_context->copier()->packHalf(dest, arg->element({0}), arg->element({1}));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to bfloat16x2: ",
                              ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateFP8x4(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::FP8:
        {
            AssertFatal(arg->valueCount() == 4,
                        "Conversion to FP8x4 requires four elements",
                        ShowValue(arg->valueCount()));
            std::vector<Register::ValuePtr> values{
                arg->element({0}), arg->element({1}), arg->element({2}), arg->element({3})};
            co_yield m_context->copier()->pack(dest, values, "Pack into FP8x4");
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to FP8x4: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateBF8x4(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::BF8:
        {
            AssertFatal(arg->valueCount() == 4,
                        "Conversion to BF8x4 requires four elements",
                        ShowValue(arg->valueCount()));
            std::vector<Register::ValuePtr> values{
                arg->element({0}), arg->element({1}), arg->element({2}), arg->element({3})};
            co_yield m_context->copier()->pack(dest, values, "Pack into BF8x4");
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to BF8x4: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateFP8(Register::ValuePtr dest,
                                                         Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
        {
            // F32 to F8 only supports packed conversion currently. But for simplicity,
            // here we convert only one value at a time. If we want to convert
            // two distinct F32 at a time, the caller of this function must ensure
            // two values (in arg) are passed in.
            // TODO: this (might) wastes registers (four FP8 can be packed in a
            // register at most)
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "Zero out register for converting F32 to FP8");
            // Note: the second input is set to 0 to ensure the destination register only
            //       contains a FP8 value (i.e., the other 24 bits are 0)
            co_yield_(Instruction(
                "v_cvt_pk_fp8_f32", {dest}, {arg, Register::Value::Literal(0)}, {}, ""));
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to FP8: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateBF8(Register::ValuePtr dest,
                                                         Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
        {
            // F32 to F8 only supports packed conversion currently. But for simplicity,
            // here we convert only one value at a time. If we want to convert
            // two distinct F32 at a time, the caller of this function must ensure
            // two values (in arg) are passed in.
            // TODO: this (might) wastes registers (four BF8 can be packed in a
            // register at most)
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "Zero out register for converting F32 to BF8");
            // Note: the second input is set to 0 to ensure the destination register only
            //       contains a BF8 value (i.e., the other 24 bits are 0)
            co_yield_(Instruction(
                "v_cvt_pk_bf8_f32", {dest}, {arg, Register::Value::Literal(0)}, {}, ""));
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to bf8: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateFP6x16(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        auto dataType = getArithDataType(arg);
        Throw<FatalError>("Unsupported datatype for convert to FP6x16 ", ShowValue(dataType));
    }

    Generator<Instruction> ConvertGenerator::generateBF6x16(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        auto dataType = getArithDataType(arg);
        Throw<FatalError>("Unsupported datatype for convert to BF6x16 ", ShowValue(dataType));
    }

    Generator<Instruction> ConvertGenerator::generateFP4x8(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        auto dataType = getArithDataType(arg);
        Throw<FatalError>("Unsupported datatype for convert to FP4x8 ", ShowValue(dataType));
    }

    Generator<Instruction> ConvertGenerator::generateInt32(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::UInt32:
            co_yield m_context->copier()->copy(dest, arg);
            break;

        case DataType::UInt64:
        case DataType::Int64:
            co_yield m_context->copier()->copy(dest, arg->subset({0}));
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to Int32: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateInt64(Register::ValuePtr dest,
                                                           Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield signExtendDWord(dest->subset({1}), arg);
            co_yield m_context->copier()->copy(dest->subset({0}), arg, "convert");
            break;
        case DataType::UInt32:
            co_yield m_context->copier()->copy(
                dest->subset({1}), Register::Value::Literal(0), "convert");
            co_yield m_context->copier()->copy(dest->subset({0}), arg, "convert");
            break;

        case DataType::UInt64:
            co_yield m_context->copier()->copy(dest, arg, "convert");
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to Int64: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateUInt32(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield m_context->copier()->copy(dest, arg, "convert");
            break;

        case DataType::UInt64:
        case DataType::Int64:
            co_yield m_context->copier()->copy(dest, arg->subset({0}), "convert");
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to UInt32: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateUInt64(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)

    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield signExtendDWord(dest->subset({1}), arg);
            co_yield m_context->copier()->copy(dest->subset({0}), arg, "convert");
            break;
        case DataType::UInt32:
            co_yield m_context->copier()->copy(
                dest->subset({1}), Register::Value::Literal(0), "convert");
            co_yield m_context->copier()->copy(dest->subset({0}), arg, "convert");
            break;

        case DataType::Int64:
            co_yield m_context->copier()->copy(dest, arg, "convert");
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to UInt64: ", ShowValue(dataType));
        }
    }

    Generator<Instruction> ConvertGenerator::generateDouble(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)

    {
        Throw<FatalError>("Convert to Double not supported");
    }

    RegisterComponentTemplateSpec(SRConvertGenerator, DataType::FP8);
    RegisterComponentTemplateSpec(SRConvertGenerator, DataType::BF8);

#define DefineSpecializedGetGeneratorSRConvert(dtype)                                             \
    template <>                                                                                   \
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SRConvert<DataType::dtype>>>            \
        GetGenerator<Expression::SRConvert<DataType::dtype>>(                                     \
            Register::ValuePtr dst,                                                               \
            Register::ValuePtr lhs,                                                               \
            Register::ValuePtr rhs,                                                               \
            Expression::SRConvert<DataType::dtype> const&)                                        \
    {                                                                                             \
        return Component::Get<BinaryArithmeticGenerator<Expression::SRConvert<DataType::dtype>>>( \
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);   \
    }

    DefineSpecializedGetGeneratorSRConvert(FP8);
    DefineSpecializedGetGeneratorSRConvert(BF8);
#undef DefineSpecializedGetSRGeneratorConvert

    template <>
    Generator<Instruction>
        SRConvertGenerator<DataType::FP8>::generate(Register::ValuePtr dest,
                                                    Register::ValuePtr lhs,
                                                    Register::ValuePtr rhs,
                                                    Expression::SRConvert<DataType::FP8> const&)
    {
        AssertFatal(lhs != nullptr && rhs != nullptr);

        auto dataType = getArithDataType(lhs);

        switch(dataType)
        {
        case DataType::Float:
        {
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "Zero out register for converting F32 to FP8");
            co_yield_(Instruction("v_cvt_sr_fp8_f32", {dest}, {lhs, rhs}, {}, ""));
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for SR convert to FP8: ", ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction>
        SRConvertGenerator<DataType::BF8>::generate(Register::ValuePtr dest,
                                                    Register::ValuePtr lhs,
                                                    Register::ValuePtr rhs,
                                                    Expression::SRConvert<DataType::BF8> const&)
    {
        AssertFatal(lhs != nullptr && rhs != nullptr);

        auto dataType = getArithDataType(lhs);

        switch(dataType)
        {
        case DataType::Float:
        {
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "Zero out register for converting F32 to BF8");
            co_yield_(Instruction("v_cvt_sr_bf8_f32", {dest}, {lhs, rhs}, {}, ""));
        }
        break;
        default:
            Throw<FatalError>("Unsupported datatype for SR convert to BF8: ", ShowValue(dataType));
        }
    }

}

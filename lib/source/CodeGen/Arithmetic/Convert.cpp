#include <rocRoller/CodeGen/Arithmetic/Convert.hpp>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Double);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Float);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Half);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Halfx2);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::BFloat16);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::BFloat16x2);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::FP8);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::BF8);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::FP8x4);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::BF8x4);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Int32);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Int64);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::UInt32);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::UInt64);

#define DefineSpecializedGetGeneratorConvert(dtype)                                            \
    template <>                                                                                \
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::dtype>>>            \
        GetGenerator<Expression::Convert<DataType::dtype>>(Register::ValuePtr dst,             \
                                                           Register::ValuePtr arg)             \
    {                                                                                          \
        return Component::Get<UnaryArithmeticGenerator<Expression::Convert<DataType::dtype>>>( \
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);     \
    }

    DefineSpecializedGetGeneratorConvert(Double);
    DefineSpecializedGetGeneratorConvert(Float);
    DefineSpecializedGetGeneratorConvert(Half);
    DefineSpecializedGetGeneratorConvert(Halfx2);
    DefineSpecializedGetGeneratorConvert(BFloat16);
    DefineSpecializedGetGeneratorConvert(BFloat16x2);
    DefineSpecializedGetGeneratorConvert(FP8);
    DefineSpecializedGetGeneratorConvert(BF8);
    DefineSpecializedGetGeneratorConvert(FP8x4);
    DefineSpecializedGetGeneratorConvert(BF8x4);
    DefineSpecializedGetGeneratorConvert(Int32);
    DefineSpecializedGetGeneratorConvert(Int64);
    DefineSpecializedGetGeneratorConvert(UInt32);
    DefineSpecializedGetGeneratorConvert(UInt64);
#undef DefineSpecializedGetGeneratorConvert

    Generator<Instruction>
        generateConvertOp(DataType dataType, Register::ValuePtr dest, Register::ValuePtr arg)
    {
#define ConvertCase(dtype)                                                    \
    case DataType::dtype:                                                     \
        co_yield generateOp<Expression::Convert<DataType::dtype>>(dest, arg); \
        break

        switch(dataType)
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

        default:
            Throw<FatalError>("Unsupported datatype conversion: ", ShowValue(dataType));
        }
#undef ConvertCase
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Float>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Half>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Halfx2>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Half:
            AssertFatal(arg->valueCount() == 2, "Conversion to Halfx2 requires two elements");
            co_yield m_context->copier()->packHalf(dest, arg->element({0}), arg->element({1}));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to halfx2: ", ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::BFloat16>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::BFloat16x2>::generate(Register::ValuePtr dest,
                                                                            Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::BFloat16:
            AssertFatal(arg->valueCount() == 2, "Conversion to Bfloat16x2 requires two elements");
            co_yield m_context->copier()->packHalf(dest, arg->element({0}), arg->element({1}));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to bfloat16x2: ",
                              ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::FP8x4>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::BF8x4>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::FP8>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::BF8>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Int32>::generate(Register::ValuePtr dest,
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

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Int64>::generate(Register::ValuePtr dest,
                                                                       Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield signExtendDWord(dest->subset({1}), arg);
            co_yield m_context->copier()->copy(dest->subset({0}), arg);
            break;
        case DataType::UInt32:
            co_yield m_context->copier()->copy(dest->subset({1}), Register::Value::Literal(0));
            co_yield m_context->copier()->copy(dest->subset({0}), arg);
            break;

        case DataType::UInt64:
            co_yield m_context->copier()->copy(dest, arg);
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to Int64: ", ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::UInt32>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield m_context->copier()->copy(dest, arg);
            break;

        case DataType::UInt64:
        case DataType::Int64:
            co_yield m_context->copier()->copy(dest, arg->subset({0}));
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to UInt32: ", ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::UInt64>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Int32:
            co_yield signExtendDWord(dest->subset({1}), arg);
            co_yield m_context->copier()->copy(dest->subset({0}), arg);
            break;
        case DataType::UInt32:
            co_yield m_context->copier()->copy(dest->subset({1}), Register::Value::Literal(0));
            co_yield m_context->copier()->copy(dest->subset({0}), arg);
            break;

        case DataType::Int64:
            co_yield m_context->copier()->copy(dest, arg);
            break;

        default:
            Throw<FatalError>("Unsupported datatype for convert to UInt64: ", ShowValue(dataType));
        }
    }

    template <>
    Generator<Instruction> ConvertGenerator<DataType::Double>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr arg)
    {
        Throw<FatalError>("Convert to Double not supported");
    }

}

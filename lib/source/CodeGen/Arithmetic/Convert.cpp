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

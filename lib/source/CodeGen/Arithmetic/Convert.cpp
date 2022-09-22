#include <rocRoller/CodeGen/Arithmetic/Convert.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Float);
    RegisterComponentTemplateSpec(ConvertGenerator, DataType::Half);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::Float>>>
        GetGenerator<Expression::Convert<DataType::Float>>(Register::ValuePtr dst,
                                                           Register::ValuePtr arg)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Convert<DataType::Float>>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::Half>>>
        GetGenerator<Expression::Convert<DataType::Half>>(Register::ValuePtr dst,
                                                          Register::ValuePtr arg)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Convert<DataType::Half>>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction>
        generateConvertOp(DataType dataType, Register::ValuePtr dest, Register::ValuePtr arg)
    {
        switch(dataType)
        {
        case DataType::Float:
            co_yield generateOp<Expression::Convert<DataType::Float>>(dest, arg);
            break;
        case DataType::Half:
            co_yield generateOp<Expression::Convert<DataType::Half>>(dest, arg);
            break;
        default:
            Throw<FatalError>("Unsupported datatype conversion");
        }
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
            co_yield generateOp<Expression::ShiftR>(
                dest->element({1}), arg, Register::Value::Literal(16u));
            co_yield_(
                Instruction("v_cvt_f32_f16", {dest->element({1})}, {dest->element({1})}, {}, ""));
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

        auto dataType = getArithDataType(arg);

        switch(dataType)
        {
        case DataType::Float:
            co_yield_(Instruction("v_cvt_f16_f32", {dest}, {arg}, {}, ""));
            break;
        default:
            Throw<FatalError>("Unsupported datatype for convert to half: ", ShowValue(dataType));
        }
    }

}

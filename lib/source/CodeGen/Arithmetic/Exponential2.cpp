#include <rocRoller/CodeGen/Arithmetic/Exponential2.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(Exponential2Generator, Register::Type::Vector, DataType::Float);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Exponential2>>
        GetGenerator<Expression::Exponential2>(Register::ValuePtr dst, Register::ValuePtr arg)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Exponential2>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest, Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        co_yield_(Instruction("v_exp_f32", {dest}, {arg}, {}, ""));
    }

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Half>::generate(
        Register::ValuePtr dest, Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        co_yield_(Instruction("v_exp_f16", {dest}, {arg}, {}, ""));
    }
}

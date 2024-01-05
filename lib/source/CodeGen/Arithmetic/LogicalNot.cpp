#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalNot.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(LogicalNotGenerator, Register::Type::Scalar, DataType::Bool);
    RegisterComponentTemplateSpec(LogicalNotGenerator, Register::Type::Scalar, DataType::Bool32);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::LogicalNot>>
        GetGenerator<Expression::LogicalNot>(Register::ValuePtr dst, Register::ValuePtr arg)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<UnaryArithmeticGenerator<Expression::LogicalNot>>(
            getContextFromValues(dst, arg), arg->regType(), arg->variableType().dataType);
    }

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dst->registerCount() == 1, "Only single-register dst currently supported");

        co_yield_(Instruction("s_xor_b32", {dst}, {arg, Register::Value::Literal(1)}, {}, ""));
    }

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool32>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dst->registerCount() == 1, "Only single-register dst currently supported");

        co_yield_(Instruction("s_xor_b32", {dst}, {arg, Register::Value::Literal(1)}, {}, ""));
    }
}

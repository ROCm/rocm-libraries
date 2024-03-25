#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalNot.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(LogicalNotGenerator, Register::Type::Scalar, DataType::Bool);
    RegisterComponentTemplateSpec(LogicalNotGenerator, Register::Type::Scalar, DataType::Bool32);
    RegisterComponentTemplateSpec(LogicalNotGenerator, Register::Type::Scalar, DataType::Bool64);

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

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool64>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        auto tmp
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Bool64, 1);

        co_yield_(Instruction("s_xor_b64", {tmp}, {arg, Register::Value::Literal(1)}, {}, ""));
        co_yield_(Instruction("s_cmp_lg_u64", {}, {tmp, Register::Value::Literal(0)}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }
}

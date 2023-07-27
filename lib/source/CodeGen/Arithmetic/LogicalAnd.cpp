#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalAnd.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(LogicalAndGenerator, Register::Type::Scalar, DataType::Bool32);
    RegisterComponentTemplateSpec(LogicalAndGenerator, Register::Type::Scalar, DataType::UInt64);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LogicalAnd>>
        GetGenerator<Expression::LogicalAnd>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::LogicalAnd>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(nullptr, lhs, rhs),
            promoteDataType(nullptr, lhs, rhs));
    }

    template <>
    Generator<Instruction> LogicalAndGenerator<Register::Type::Scalar, DataType::Bool32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        auto tmp
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Raw32, 1);

        co_yield_(Instruction("s_and_b32", {tmp}, {lhs, rhs}, {}, ""));
        co_yield_(Instruction("s_cmp_lg_u32", {}, {tmp, Register::Value::Literal(0)}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction> LogicalAndGenerator<Register::Type::Scalar, DataType::UInt64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        auto tmp
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt64, 1);

        co_yield_(Instruction("s_and_b64", {tmp}, {lhs, rhs}, {}, ""));
        co_yield_(Instruction("s_cmp_lg_u64", {}, {tmp, Register::Value::Literal(0)}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }
}

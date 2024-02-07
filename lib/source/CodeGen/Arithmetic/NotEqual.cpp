#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/NotEqual.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Vector, DataType::Int32);
    // Covered by `match` as they are same instruction Register::Type::Scalar, DataType::Int32
    // Covered by `match` as they are same instruction Register::Type::Vector, DataType::Int32
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::NotEqual>>
        GetGenerator<Expression::NotEqual>(Register::ValuePtr dst,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::NotEqual>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(nullptr, lhs, rhs),
            promoteDataType(nullptr, lhs, rhs));
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        co_yield_(Instruction("s_cmp_lg_u32", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ne_i32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        co_yield_(Instruction("s_cmp_lg_u64", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ne_i64", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_neq_f32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> NotEqualGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_neq_f64", {dst}, {lhs, rhs}, {}, ""));
    }
}

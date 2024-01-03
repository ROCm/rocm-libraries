#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/NotEqual.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(NotEqualGenerator, Register::Type::Vector, DataType::Int32);
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

        Register::ValuePtr tmp_rhs;
        co_yield m_context->copier()->ensureType(tmp_rhs, rhs, Register::Type::Vector);

        if(dst->registerCount() == 2)
        {
            co_yield_(Instruction("v_cmp_ne_i32", {dst}, {lhs, tmp_rhs}, {}, ""));
        }
        else if(dst->registerCount() == 1)
        {
            auto tmp_dst = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::UInt64, 1);
            co_yield tmp_dst->allocate();
            co_yield_(Instruction("v_cmp_ne_i32", {tmp_dst}, {lhs, tmp_rhs}, {}, ""));
            co_yield m_context->copier()->copy(dst, tmp_dst->subset({0}), "");
        }
        else
        {
            Throw<FatalError>(
                "Unsupported dst", ShowValue(dst->registerCount()), ShowValue(dst->regType()));
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

        Register::ValuePtr tmp;
        co_yield m_context->copier()->ensureType(tmp, rhs, Register::Type::Vector);

        co_yield_(Instruction("v_cmp_ne_i64", {dst}, {lhs, tmp}, {}, ""));
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

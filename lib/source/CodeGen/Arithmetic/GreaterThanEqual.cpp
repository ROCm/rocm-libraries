#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/GreaterThanEqual.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Scalar,
                                  DataType::Int32);
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Vector,
                                  DataType::Int32);
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Scalar,
                                  DataType::Int64);
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Vector,
                                  DataType::Int64);
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Vector,
                                  DataType::Float);
    RegisterComponentTemplateSpec(GreaterThanEqualGenerator,
                                  Register::Type::Vector,
                                  DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>
        GetGenerator<Expression::GreaterThanEqual>(Register::ValuePtr dst,
                                                   Register::ValuePtr lhs,
                                                   Register::ValuePtr rhs)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(nullptr, lhs, rhs),
            promoteDataType(nullptr, lhs, rhs));
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int32>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_ge_i32", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
        }
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int32>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_i32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int64>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto vTemp = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield vTemp->allocate();

        co_yield m_context->copier()->copy(vTemp, rhs, "");

        co_yield_(Instruction("v_cmp_ge_i64", {dst}, {lhs, vTemp}, {}, ""));
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int64>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_i64", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Vector, DataType::Float>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_f32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        GreaterThanEqualGenerator<Register::Type::Vector, DataType::Double>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_f64", {dst}, {lhs, rhs}, {}, ""));
    }
}

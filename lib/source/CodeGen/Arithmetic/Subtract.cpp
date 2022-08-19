#include <rocRoller/CodeGen/Arithmetic/Subtract.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Scalar, DataType::UInt32);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Vector, DataType::UInt32);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(SubtractGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Subtract>>
        GetGenerator<Expression::Subtract>(Register::ValuePtr dst,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Subtract>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_sub_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_sub_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "s_sub_u32", {dest->subset({0})}, {l0, r0}, {}, "least significant half; sets scc"));

        co_yield_(Instruction(
            "s_subb_u32", {dest->subset({1})}, {l1, r1}, {}, "most significant half; uses scc"));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        auto borrow = m_context->getVCC();

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "v_sub_co_u32", {dest->subset({0}), borrow}, {l0, r0}, {}, "least significant half"));

        co_yield_(Instruction("v_subb_co_u32",
                              {dest->subset({1}), borrow},
                              {l1, r1, borrow},
                              {},
                              "most significant half"));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto nrhs = rhs->negate();

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, nrhs}, {}, ""));
    }

}

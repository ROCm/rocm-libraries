#include <rocRoller/CodeGen/Arithmetic/Add.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::UInt32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::UInt32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Halfx2);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Add>> GetGenerator<Expression::Add>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Add>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_add_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield swapIfRHSLiteral(lhs, rhs);

        co_yield_(Instruction("v_add_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_add_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield swapIfRHSLiteral(lhs, rhs);

        co_yield_(Instruction("v_add_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(
            Instruction(
                "s_add_u32", {dest->subset({0})}, {l0, r0}, {}, "least significant half; sets scc")
                .lock(Scheduling::Dependency::SCC));

        co_yield_(
            Instruction(
                "s_addc_u32", {dest->subset({1})}, {l1, r1}, {}, "most significant half; uses scc")
                .unlock());
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        if(r0->regType() == Register::Type::Literal)
            std::swap(l0, r0);

        if(r0->regType() == Register::Type::Scalar)
        {
            co_yield moveToVGPR(r0);
        }

        if(l1->regType() == Register::Type::Scalar
           || (l1->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(l1)))
        {
            co_yield moveToVGPR(l1);
        }

        if(r1->regType() == Register::Type::Scalar
           || (r1->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(r1)))
        {
            co_yield moveToVGPR(r1);
        }

        auto carry = m_context->getVCC();

        co_yield Register::AllocateIfNeeded(dest);

        // LOCK: Carry Register needed
        co_yield_(
            Instruction(
                "v_add_co_u32", {dest->subset({0}), carry}, {l0, r0}, {}, "least significant half")
                .lock(Scheduling::Dependency::VCC));

        co_yield_(Instruction("v_addc_co_u32",
                              {dest->subset({1}), carry},
                              {l1, r1, carry},
                              {},
                              "most significant half")
                      .unlock());
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Halfx2>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_pk_add_f16", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield swapIfRHSLiteral(lhs, rhs);

        co_yield_(Instruction("v_add_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dest, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield swapIfRHSLiteral(lhs, rhs);

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, rhs}, {}, ""));
    }

}

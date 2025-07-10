/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/AddInstruction.hpp>
#include <rocRoller/CodeGen/Arithmetic/Add.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::M0, DataType::UInt32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::UInt32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::UInt32);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Half);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Halfx2);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::BFloat16);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(AddGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Add>>
        GetGenerator<Expression::Add>(Register::ValuePtr dst,
                                      Register::ValuePtr lhs,
                                      Register::ValuePtr rhs,
                                      Expression::Add const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Add>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Scalar, DataType::Int32>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr rhs,
                                                                        Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield ScalarAddInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Int32>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr rhs,
                                                                        Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield m_context->copier()->ensureTypeCommutative(
            {Register::Type::Vector, Register::Type::Constant},
            lhs,
            {Register::Type::Vector, Register::Type::Constant},
            rhs);

        auto const& arch = m_context->targetArchitecture();
        auto const& gpu  = arch.target();

        if(arch.HasCapability(GPUCapability::HasExplicitNC))
        {
            co_yield_(Instruction("v_add_nc_i32", {dest}, {lhs, rhs}, {}, ""));
        }
        else if(gpu.isCDNAGPU())
        {
            co_yield_(Instruction("v_add_i32", {dest}, {lhs, rhs}, {}, ""));
        }
        else
        {
            Throw<FatalError>(fmt::format("AddGenerator<{}, {}> not implemented for {}.\n",
                                          toString(Register::Type::Vector),
                                          toString(DataType::Int32),
                                          gpu.toString()));
        }
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Scalar, DataType::UInt32>::generate(Register::ValuePtr dest,
                                                                         Register::ValuePtr lhs,
                                                                         Register::ValuePtr rhs,
                                                                         Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield ScalarAddUInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::UInt32>::generate(Register::ValuePtr dest,
                                                                         Register::ValuePtr lhs,
                                                                         Register::ValuePtr rhs,
                                                                         Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield m_context->copier()->ensureTypeCommutative(
            {Register::Type::Vector, Register::Type::Literal}, lhs, {Register::Type::Vector}, rhs);

        co_yield VectorAddUInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::M0, DataType::UInt32>::generate(Register::ValuePtr dest,
                                                                     Register::ValuePtr lhs,
                                                                     Register::ValuePtr rhs,
                                                                     Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield ScalarAddUInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Scalar, DataType::Int64>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr rhs,
                                                                        Expression::Add const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        co_yield(Instruction::Lock(Scheduling::Dependency::SCC, "Start of Int64 add, locking SCC"));

        co_yield ScalarAddUInt32(m_context, dest->subset({0}), l0, r0);
        co_yield ScalarAddUInt32CarryInOut(m_context, dest->subset({1}), l1, r1);

        co_yield(Instruction::Unlock("End of Int64 add, unlocking SCC"));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Int64>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr rhs,
                                                                        Expression::Add const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        if(r0->regType() == Register::Type::Literal)
        {
            std::swap(l0, r0);
        }

        if(r1->regType() == Register::Type::Literal)
        {
            std::swap(l1, r1);
        }

        if(r0->regType() == Register::Type::Scalar || r0->regType() == Register::Type::Literal)
        {
            co_yield moveToVGPR(r0);
        }

        if(r1->regType() == Register::Type::Scalar || r1->regType() == Register::Type::Literal)
        {
            co_yield moveToVGPR(r1);
        }

        if(l0->regType() == Register::Type::Scalar)
        {
            co_yield moveToVGPR(l0);
        }

        // v_addc_co_u32 can only accept inline constant literals
        if(l1->regType() == Register::Type::Scalar
           || (l1->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(l1)))
        {
            co_yield moveToVGPR(l1);
        }

        bool useVCC = (l0->regType() == Register::Type::Literal
                       && !m_context->targetArchitecture().isSupportedConstantValue(l0))
                      || (l1->regType() == Register::Type::Literal
                          && !m_context->targetArchitecture().isSupportedConstantValue(l1));

        if(useVCC)
            co_yield(
                Instruction::Lock(Scheduling::Dependency::VCC, "Start of Int64 add, locking VCC"));

        auto carry
            = useVCC ? m_context->getVCC() : Register::Value::WavefrontPlaceholder(m_context, 1);

        co_yield VectorAddUInt32CarryOut(
            m_context, dest->subset({0}), carry, l0, r0, "least significant half");
        co_yield VectorAddUInt32CarryInOut(
            m_context, dest->subset({1}), carry, carry, l1, r1, "most significant half");

        if(useVCC)
            co_yield(Instruction::Unlock("End of Int64 add, Unlocking VCC."));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Halfx2>::generate(Register::ValuePtr dest,
                                                                         Register::ValuePtr lhs,
                                                                         Register::ValuePtr rhs,
                                                                         Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_pk_add_f16", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Half>::generate(Register::ValuePtr dest,
                                                                       Register::ValuePtr lhs,
                                                                       Register::ValuePtr rhs,
                                                                       Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_add_f16", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::BFloat16>::generate(Register::ValuePtr dest,
                                                                           Register::ValuePtr lhs,
                                                                           Register::ValuePtr rhs,
                                                                           Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto lit16 = Register::Value::Literal(16u);

        co_yield generateOp<Expression::ShiftL>(lhs, lhs, lit16);
        co_yield generateOp<Expression::ShiftL>(rhs, rhs, lit16);

        co_yield_(Instruction("v_add_f32", {dest}, {lhs, rhs}, {}, ""));
        co_yield generateOp<Expression::LogicalShiftR>(dest, dest, lit16);

        co_yield generateOp<Expression::LogicalShiftR>(lhs, lhs, lit16);
        co_yield generateOp<Expression::LogicalShiftR>(rhs, rhs, lit16);
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Float>::generate(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr rhs,
                                                                        Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield m_context->copier()->ensureTypeCommutative(
            {Register::Type::Vector, Register::Type::Constant}, lhs, {Register::Type::Vector}, rhs);

        co_yield_(Instruction("v_add_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        AddGenerator<Register::Type::Vector, DataType::Double>::generate(Register::ValuePtr dest,
                                                                         Register::ValuePtr lhs,
                                                                         Register::ValuePtr rhs,
                                                                         Expression::Add const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield m_context->copier()->ensureTypeCommutative(
            {Register::Type::Vector, Register::Type::Constant},
            lhs,
            {Register::Type::Vector, Register::Type::Constant},
            rhs);

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, rhs}, {}, ""));
    }

}

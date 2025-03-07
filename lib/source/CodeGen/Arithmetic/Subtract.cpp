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

#include <rocRoller/CodeGen/Arithmetic/Subtract.hpp>
#include <rocRoller/CodeGen/SubInstruction.hpp>
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
                                           Register::ValuePtr rhs,
                                           Expression::Subtract const&)
    {
        AssertFatal(dst != nullptr, "Null destination");
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Subtract>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(dst, lhs, rhs),
            promoteDataType(dst, lhs, rhs));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield ScalarSubInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto gpu = m_context->targetArchitecture().target();
        if(gpu.isCDNAGPU())
        {
            co_yield_(Instruction("v_sub_i32", {dest}, {lhs, rhs}, {}, ""));
        }
        else if(gpu.isRDNAGPU())
        {
            co_yield_(Instruction("v_sub_nc_i32", {dest}, {lhs, rhs}, {}, ""));
        }
        else
        {
            Throw<FatalError>(std::format("SubtractGenerator<{}, {}> not implemented for {}\n",
                                          toString(Register::Type::Vector),
                                          toString(DataType::Int32),
                                          gpu.toString()));
        }
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield ScalarSubUInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield VectorSubUInt32(m_context, dest, lhs, rhs);
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsScalar(l0, l1, lhs);
        co_yield get2DwordsScalar(r0, r1, rhs);

        co_yield(Instruction::Lock(Scheduling::Dependency::SCC, "Start of Int64 sub, locking SCC"));

        co_yield ScalarSubUInt32(
            m_context, dest->subset({0}), l0, r0, "least significant half; sets scc");
        co_yield ScalarSubUInt32CarryInOut(
            m_context, dest->subset({1}), l1, r1, "most significant half; uses scc");

        co_yield(Instruction::Unlock("End of Int64 sub, unlocking SCC"));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        co_yield describeOpArgs("dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2DwordsVector(l0, l1, lhs);
        co_yield get2DwordsVector(r0, r1, rhs);

        auto borrow = m_context->getVCC();

        co_yield(Instruction::Lock(Scheduling::Dependency::VCC, "Start of Int64 sub, locking VCC"));

        co_yield VectorSubUInt32CarryOut(
            m_context, dest->subset({0}), l0, r0, "least significant half");
        co_yield VectorSubUInt32CarryInOut(
            m_context, dest->subset({1}), l1, r1, "most significant half");

        co_yield(Instruction::Unlock("End of Int64 sub, unlocking VCC"));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> SubtractGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dest,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Subtract const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto nrhs = rhs->negate();

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, nrhs}, {}, ""));
    }

}

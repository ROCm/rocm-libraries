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

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/Equal.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Vector, DataType::Int32);
    // Covered by `match` as they are same instruction Register::Type::Scalar, DataType::Int32
    // Covered by `match` as they are same instruction Register::Type::Vector, DataType::Int32
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(EqualGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Equal>>
        GetGenerator<Expression::Equal>(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::Equal const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::Equal>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(nullptr, lhs, rhs),
            promoteDataType(nullptr, lhs, rhs));
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Scalar, DataType::Int32>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        // note s_cmp_eq_i32 and s_cmp_eq_u32 are same op-codes, both are
        // provided for symmetry
        co_yield_(Instruction("s_cmp_eq_i32", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Int32>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_i32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Scalar, DataType::Int64>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        co_yield_(Instruction("s_cmp_eq_u64", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Int64>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_i64", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Float>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_f32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Double>::generate(Register::ValuePtr dst,
                                                                           Register::ValuePtr lhs,
                                                                           Register::ValuePtr rhs,
                                                                           Expression::Equal const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_f64", {dst}, {lhs, rhs}, {}, ""));
    }
}

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
        GetGenerator<Expression::LogicalNot>(Register::ValuePtr dst,
                                             Register::ValuePtr arg,
                                             Expression::LogicalNot const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<UnaryArithmeticGenerator<Expression::LogicalNot>>(
            getContextFromValues(dst, arg), arg->regType(), arg->variableType().dataType);
    }

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::LogicalNot const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dst->registerCount() == 1, "Only single-register dst currently supported");

        co_yield_(Instruction("s_xor_b32", {dst}, {arg, Register::Value::Literal(1)}, {}, ""));
    }

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool32>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::LogicalNot const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dst->registerCount() == 1, "Only single-register dst currently supported");

        co_yield_(Instruction("s_xor_b32", {dst}, {arg, Register::Value::Literal(1)}, {}, ""));
    }

    template <>
    Generator<Instruction> LogicalNotGenerator<Register::Type::Scalar, DataType::Bool64>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::LogicalNot const&)
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

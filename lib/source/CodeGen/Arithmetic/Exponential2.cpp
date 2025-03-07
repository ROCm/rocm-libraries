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

#include <rocRoller/CodeGen/Arithmetic/Exponential2.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(Exponential2Generator, Register::Type::Vector, DataType::Float);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Exponential2>>
        GetGenerator<Expression::Exponential2>(Register::ValuePtr dst,
                                               Register::ValuePtr arg,
                                               Expression::Exponential2 const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Exponential2>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dest, Register::ValuePtr arg, Expression::Exponential2 const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        co_yield_(Instruction("v_exp_f32", {dest}, {arg}, {}, ""));
    }

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Half>::generate(
        Register::ValuePtr dest, Register::ValuePtr arg, Expression::Exponential2 const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        co_yield_(Instruction("v_exp_f16", {dest}, {arg}, {}, ""));
    }
}

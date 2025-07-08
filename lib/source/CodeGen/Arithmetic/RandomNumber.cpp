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
#include <rocRoller/CodeGen/Arithmetic/RandomNumber.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(RandomNumberGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::RandomNumber>>
        GetGenerator<Expression::RandomNumber>(Register::ValuePtr dst,
                                               Register::ValuePtr arg,
                                               Expression::RandomNumber const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::RandomNumber>>(
            getContextFromValues(dst, arg), arg->regType(), arg->variableType().dataType);
    }

    Generator<Instruction> RandomNumberGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr seed,
                                                           Expression::RandomNumber const&)
    {
        AssertFatal(seed != nullptr);
        AssertFatal(dest != nullptr);
        co_yield_(Instruction("v_prng_b32", {dest}, {seed}, {}, "Generate a random number"));
    }
}

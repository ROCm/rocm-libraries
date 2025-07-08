/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
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

#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{
    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::MultiplyAdd>>
        GetGenerator<Expression::MultiplyAdd>(Register::ValuePtr dst,
                                              Register::ValuePtr a,
                                              Register::ValuePtr x,
                                              Register::ValuePtr y,
                                              Expression::MultiplyAdd const&);

    struct MultiplyAddGenerator : public TernaryArithmeticGenerator<Expression::MultiplyAdd>
    {
        MultiplyAddGenerator(ContextPtr c)
            : TernaryArithmeticGenerator<Expression::MultiplyAdd>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MultiplyAddGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dest,
                                        Register::ValuePtr a,
                                        Register::ValuePtr x,
                                        Register::ValuePtr y,
                                        Expression::MultiplyAdd const&);

        static const std::string Name;
    };
}

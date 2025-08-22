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

#pragma once

#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>

namespace rocRoller
{
    // Template to hold the actual generate implementation
    // Makes partial specialization of the template feasible
    template <bool IS_INTEGRAL, bool DO_SIGNED>
    class BFEGeneratorBase
    {
    public:
        // Method to generate instructions
        static Generator<Instruction> generate(Register::ValuePtr                 dst,
                                               Register::ValuePtr                 arg,
                                               Expression::BitFieldExtract const& expr);
    };

    template <DataType DATATYPE>
    inline Generator<Instruction> BitFieldExtractGenerator<DATATYPE>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const& expr)
    {
        if(Expression::getComment(expr) != "")
        {
            co_yield Instruction::Comment(Expression::getComment(expr));
        }
        else
        {
            co_yield Instruction::Comment(concatenate("BitFieldExtract<",
                                                      static_cast<int>(expr.offset),
                                                      ",",
                                                      static_cast<int>(expr.width),
                                                      ">(",
                                                      arg->description(),
                                                      ")"));
        }
        co_yield BFEGeneratorBase<EnumTypeInfo<DATATYPE>::IsIntegral,
                                  EnumTypeInfo<DATATYPE>::IsIntegral
                                      && EnumTypeInfo<DATATYPE>::IsSigned>::generate(dst,
                                                                                     arg,
                                                                                     expr);
    }
}

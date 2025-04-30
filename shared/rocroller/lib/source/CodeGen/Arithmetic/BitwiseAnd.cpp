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
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseAndGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseAnd>>
        GetGenerator<Expression::BitwiseAnd>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             Expression::BitwiseAnd const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseAnd>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseAndGenerator::generate(Register::ValuePtr dest,
                                                         Register::ValuePtr lhs,
                                                         Register::ValuePtr rhs,
                                                         Expression::BitwiseAnd const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto destNumBits = DataTypeInfo::Get(dest->variableType()).elementBits;
        auto lhsNumBits  = DataTypeInfo::Get(lhs->variableType()).elementBits;
        auto rhsNumBits  = DataTypeInfo::Get(rhs->variableType()).elementBits;

        auto elementBits = std::max({destNumBits, lhsNumBits, rhsNumBits});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementBits <= 32u)
            {
                co_yield_(Instruction("s_and_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction("s_and_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseAnd operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            co_yield swapIfRHSLiteral(lhs, rhs);

            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_and_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                Register::ValuePtr l0, l1, r0, r1;
                if(lhs->regType() == Register::Type::Scalar)
                {
                    co_yield get2DwordsScalar(l0, l1, lhs);
                }
                else
                {
                    co_yield get2DwordsVector(l0, l1, lhs);
                }
                co_yield get2DwordsVector(r0, r1, rhs);

                co_yield_(Instruction("v_and_b32", {dest->subset({0})}, {l0, r0}, {}, ""));
                co_yield_(Instruction("v_and_b32", {dest->subset({1})}, {l1, r1}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseAnd operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseAnd operation: ",
                              ShowValue(dest->regType()));
        }
    }
}

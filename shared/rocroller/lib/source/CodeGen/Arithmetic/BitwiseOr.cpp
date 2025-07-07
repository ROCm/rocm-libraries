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
#include <rocRoller/CodeGen/Arithmetic/BitwiseOr.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseOrGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseOr>>
        GetGenerator<Expression::BitwiseOr>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr rhs,
                                            Expression::BitwiseOr const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseOr>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseOrGenerator::generate(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs,
                                                        Expression::BitwiseOr const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto elementBits = std::max({DataTypeInfo::Get(dest->variableType()).elementBits,
                                     DataTypeInfo::Get(lhs->variableType()).elementBits,
                                     DataTypeInfo::Get(rhs->variableType()).elementBits});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementBits <= 32u)
            {
                co_yield_(Instruction("s_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction("s_or_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseOr operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            co_yield m_context->copier()->ensureTypeCommutative(
                {Register::Type::Vector, Register::Type::Literal},
                lhs,
                {Register::Type::Vector},
                rhs);

            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseOr operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseOr operation: ",
                              ShowValue(dest->regType()));
        }
    }
}

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

#include <rocRoller/CodeGen/Arithmetic/BitwiseNegate.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponent(BitwiseNegateGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitwiseNegate>>
        GetGenerator<Expression::BitwiseNegate>(Register::ValuePtr dst,
                                                Register::ValuePtr arg,
                                                Expression::BitwiseNegate const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::BitwiseNegate>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseNegateGenerator::generate(Register::ValuePtr dest,
                                                            Register::ValuePtr arg,
                                                            Expression::BitwiseNegate const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        auto elementBits = std::max(DataTypeInfo::Get(dest->variableType()).elementBits,
                                    DataTypeInfo::Get(arg->variableType()).elementBits);

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementBits <= 32u)
                co_yield_(Instruction("s_not_b32", {dest}, {arg}, {}, ""));
            else if(elementBits == 64u)
                co_yield_(Instruction("s_not_b64", {dest}, {arg}, {}, ""));
            else
                Throw<FatalError>("Unsupported elementBits for bitwiseNegate operation:: ",
                                  ShowValue(elementBits));
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_not_b32", {dest}, {arg}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(
                    Instruction("v_not_b32", {dest->subset({0})}, {arg->subset({0})}, {}, ""));
                co_yield_(
                    Instruction("v_not_b32", {dest->subset({1})}, {arg->subset({1})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseNegate operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseNegate operation: ",
                              ShowValue(dest->regType()));
        }
    }
}

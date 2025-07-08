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
#include <rocRoller/CodeGen/Arithmetic/ShiftL.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftLGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftL>>
        GetGenerator<Expression::ShiftL>(Register::ValuePtr dst,
                                         Register::ValuePtr lhs,
                                         Register::ValuePtr rhs,
                                         Expression::ShiftL const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::ShiftL>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> ShiftLGenerator::generate(Register::ValuePtr dest,
                                                     Register::ValuePtr value,
                                                     Register::ValuePtr shiftAmount,
                                                     Expression::ShiftL const&)
    {
        co_yield describeOpArgs("dest", dest, "value", value, "shiftAmount", shiftAmount);

        AssertFatal(dest != nullptr);
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        auto toShift = shiftAmount->regType() == Register::Type::Literal ? shiftAmount
                                                                         : shiftAmount->subset({0});

        auto resultSize = dest->variableType().getElementSize();
        auto inputSize  = value->variableType().getElementSize();

        auto resultDWords = CeilDivide(resultSize, 4ul);
        auto inputDWords  = CeilDivide(inputSize, 4ul);

        // We only want to do this conversion if there are more DWords in the output than the
        // input.  For example if we are using the shift to convert between Halfx2 and Half, both
        // use only 1 DWord.

        AssertFatal(resultDWords >= inputDWords,
                    ShowValue(dest->variableType()),
                    ShowValue(value->variableType()));

        if(resultDWords > inputDWords)
        {
            AssertFatal(!dest->intersects(shiftAmount),
                        "Destination intersecting with shift amount not yet supported.");
            co_yield generateConvertOp(dest->variableType().getArithmeticType(), dest, value);
            value = dest;
        }

        if(dest->regType() == Register::Type::Scalar)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("s_lshl_b32", {dest}, {value, toShift}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("s_lshl_b64", {dest}, {value, toShift}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("v_lshlrev_b32", {dest}, {toShift, value}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("v_lshlrev_b64", {dest}, {toShift, value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for shift left operation: ",
                              ShowValue(dest->regType()));
        }
    }
}

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

#include <rocRoller/CodeGen/Arithmetic/AddShiftL.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(AddShiftLGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::AddShiftL>>
        GetGenerator<Expression::AddShiftL>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr rhs,
                                            Register::ValuePtr shiftAmount,
                                            Expression::AddShiftL const&)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::AddShiftL>>(
            getContextFromValues(dst, lhs, rhs, shiftAmount),
            dst->regType(),
            dst->variableType().dataType);
    }

    Generator<Instruction> AddShiftLGenerator::generate(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs,
                                                        Register::ValuePtr shiftAmount,
                                                        Expression::AddShiftL const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        if(dest->regType() == Register::Type::Vector
           && (dest->variableType() == DataType::UInt32 || dest->variableType() == DataType::Int32))
        {
            auto toShift = shiftAmount->regType() == Register::Type::Literal
                               ? shiftAmount
                               : shiftAmount->subset({0});

            co_yield_(Instruction("v_add_lshl_u32", {dest}, {lhs, rhs, toShift}, {}, ""));
        }
        else
        {
            co_yield generateOp<Expression::Add>(dest, lhs, rhs);
            co_yield generateOp<Expression::ShiftL>(dest, dest, shiftAmount);
        }
    }
}

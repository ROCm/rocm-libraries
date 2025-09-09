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

#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr fromKernelArgument(AssemblyKernelArgument const& arg)
        {
            return std::make_shared<Expression>(std::make_shared<AssemblyKernelArgument>(arg));
        }

        ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(MultiplyHigh{a, b});
        }

        ExpressionPtr multiplyAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(MultiplyAdd{a, b, c});
        }

        ExpressionPtr addShiftL(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(AddShiftL{a, b, c});
        }

        ExpressionPtr shiftLAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(ShiftLAdd{a, b, c});
        }

        ExpressionPtr conditional(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(Conditional{a, b, c});
        }

        ExpressionPtr magicMultiple(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicMultiple{a});
        }

        ExpressionPtr magicShifts(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicShifts{a});
        }

        ExpressionPtr magicShiftAndSign(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicShiftAndSign{a});
        }

        ExpressionPtr bfe(ExpressionPtr a, uint8_t offset, uint8_t width)
        {
            return std::make_shared<Expression>(
                BitFieldExtract{{.arg{a}}, DataType::None, offset, width});
        }

        ExpressionPtr bfe(DataType dt, ExpressionPtr a, uint8_t offset, uint8_t width)
        {
            return std::make_shared<Expression>(BitFieldExtract{{.arg{a}}, dt, offset, width});
        }

        ExpressionPtr dataFlowTag(int tag, Register::Type t, VariableType v)
        {
            return std::make_shared<Expression>(DataFlowTag{tag, t, v});
        }

        ExpressionPtr positionalArgument(int slot, Register::Type t, VariableType v)
        {
            return std::make_shared<Expression>(PositionalArgument{slot, t, v});
        }
    }
}

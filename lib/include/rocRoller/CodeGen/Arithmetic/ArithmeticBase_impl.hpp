/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#include <functional>
#include <memory>

#include "ArithmeticBase.hpp"

#include "../../InstructionValues/Register.hpp"

namespace rocRoller
{
    template <Register::Type RegType, DataType DType, PointerType PType>
    ArithmeticBase<RegType, DType, PType>::ArithmeticBase(std::shared_ptr<Context> context)
        : Arithmetic(context)
    {
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    DataType ArithmeticBase<RegType, DType, PType>::dataType()
    {
        return DType;
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Register::Type ArithmeticBase<RegType, DType, PType>::registerType()
    {
        return RegType;
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Register::ValuePtr ArithmeticBase<RegType, DType, PType>::placeholder()
    {
        return Register::Value::Placeholder(m_context, RegType, {DType, PType}, 1);
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Generator<Instruction>
        ArithmeticBase<RegType, DType, PType>::getDwords(std::vector<Register::ValuePtr>& dwords,
                                                         Register::ValuePtr               input)
    {
        assert(input->regType() == RegType);
        assert(input->variableType() == VariableType(DType, PType));

        auto count = input->registerCount();

        dwords.clear();
        dwords.reserve(count);
        for(int i = 0; i < count; i++)
        {
            dwords.push_back(input->subset({i}));
        }

        co_return;
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Generator<Instruction> ArithmeticBase<RegType, DType, PType>::negate(Register::ValuePtr dest,
                                                                         Register::ValuePtr src)
    {
        co_yield sub(dest, Register::Value::Literal(0), src);
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Generator<Instruction>
        ArithmeticBase<RegType, DType, PType>::describeOpArgs(std::string const& opName,
                                                              std::string const& argName0,
                                                              Register::ValuePtr arg0,
                                                              std::string const& argName1,
                                                              Register::ValuePtr arg1)
    {
        co_yield describeOpArgs(opName, argName0, arg0, argName1, arg1, "", nullptr);
    }

    template <Register::Type RegType, DataType DType, PointerType PType>
    Generator<Instruction>
        ArithmeticBase<RegType, DType, PType>::describeOpArgs(std::string const& opName,
                                                              std::string const& argName0,
                                                              Register::ValuePtr arg0,
                                                              std::string const& argName1,
                                                              Register::ValuePtr arg1,
                                                              std::string const& argName2,
                                                              Register::ValuePtr arg2)
    {
        auto        opDesc = name() + " " + opName + ": ";
        std::string indent(opDesc.size(), ' ');

        co_yield Instruction::Comment(
            concatenate(opDesc, argName0, " (", arg0->description(), ") = "));
        co_yield Instruction::Comment(
            concatenate(indent, argName1, " (", arg1->description(), ") ", opName));

        if(arg2)
            co_yield Instruction::Comment(
                concatenate(indent, argName2, " (", arg2->description(), ")"));
    }
}

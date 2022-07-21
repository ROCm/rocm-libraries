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

#include "../../DataTypes/DataTypes.hpp"
#include "../Arithmetic.hpp"

namespace rocRoller
{

    template <Register::Type RegType, DataType DType, PointerType PType>
    class ArithmeticBase : public Arithmetic
    {
    public:
        ArithmeticBase(std::shared_ptr<Context> context);

        virtual DataType       dataType() final override;
        virtual Register::Type registerType() final override;

        virtual Register::ValuePtr placeholder() final override;

        virtual Generator<Instruction> getDwords(std::vector<Register::ValuePtr>& dwords,
                                                 Register::ValuePtr               inputs) override;

        virtual Generator<Instruction> negate(Register::ValuePtr dest,
                                              Register::ValuePtr src) override;

        /**
         * Generates a comment to describe an operation.
         */
        Generator<Instruction> describeOpArgs(std::string const& opName,
                                              std::string const& argName0,
                                              Register::ValuePtr arg0,
                                              std::string const& argName1,
                                              Register::ValuePtr arg1);

        Generator<Instruction> describeOpArgs(std::string const& opName,
                                              std::string const& argName0,
                                              Register::ValuePtr arg0,
                                              std::string const& argName1,
                                              Register::ValuePtr arg1,
                                              std::string const& argName2,
                                              Register::ValuePtr arg2);
    };

}

#include "ArithmeticBase_impl.hpp"

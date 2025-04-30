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

#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/CommandArguments_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>

namespace rocRoller
{
    std::string   toString(ArgumentType);
    std::ostream& operator<<(std::ostream&, ArgumentType);

    class CommandArguments
    {
    public:
        CommandArguments() = delete;
        CommandArguments(ArgumentOffsetMapPtr, int);

        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, int dimension, T value);
        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, T value);

        RuntimeArguments runtimeArguments() const;

    private:
        ArgumentOffsetMapPtr m_argOffsetMapPtr;
        KernelArguments      m_kArgs;
    };
}

#include <rocRoller/Operations/CommandArguments_impl.hpp>

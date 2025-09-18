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

#include <rocRoller/Operations/CommandArguments.hpp>

namespace rocRoller
{
    inline std::string toString(ArgumentType argType)
    {
        switch(argType)
        {
        case ArgumentType::Value:
            return "Value";
        case ArgumentType::Limit:
            return "Limit";
        case ArgumentType::Size:
            return "Size";
        case ArgumentType::Stride:
            return "Stride";
        default:
            break;
        }
        throw std::runtime_error("Invalid ArgumentType");
    }

    inline std::ostream& operator<<(std::ostream& stream, ArgumentType argType)
    {
        return stream << toString(argType);
    }

    inline CommandArguments::CommandArguments(ArgumentOffsetMapPtr argOffsetMapPtr, int bytes)
        : m_argOffsetMapPtr(argOffsetMapPtr)
        , m_kArgs(false, bytes)
    {
    }

    template <CCommandArgumentValue T>
    void CommandArguments::setArgument(Operations::OperationTag op,
                                       ArgumentType             argType,
                                       int                      dim,
                                       T                        value)
    {
        auto itr = m_argOffsetMapPtr->find(std::make_tuple(op, argType, dim));
        AssertFatal(itr != m_argOffsetMapPtr->end(),
                    "Command argument not found.",
                    ShowValue(op),
                    ShowValue(argType),
                    ShowValue(dim));

        m_kArgs.writeValue(itr->second, value);
    }

    template <CCommandArgumentValue T>
    void CommandArguments::setArgument(Operations::OperationTag op, ArgumentType argType, T value)
    {
        setArgument(op, argType, -1, value);
    }

    inline RuntimeArguments CommandArguments::runtimeArguments() const
    {
        return m_kArgs.runtimeArguments();
    }
}

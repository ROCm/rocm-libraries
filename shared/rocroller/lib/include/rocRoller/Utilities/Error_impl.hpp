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

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    template <typename Ta, typename Tb, typename... Ts>
    Error::Error(Ta const& first, Tb const& second, Ts const&... vals)
        : Error(concatenate(first, second, vals...))
    {
    }

    // Implementation for AssertError / AssertFatal / AssertRecoverable.
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void ThrowTagged(std::source_location location,
                                  const char*          exceptionTag,
                                  const char*          conditionText,
                                  Ts const&... message)
    {
        auto prefix = concatenate(GetBaseFileName(location.file_name()),
                                  ":",
                                  location.line(),
                                  ": ",
                                  exceptionTag,
                                  "(",
                                  conditionText,
                                  ")\n");

        auto fullMessage = concatenate(prefix, message...);

        bool var = Error::BreakOnThrow();
        if(var)
        {
            std::cerr << fullMessage << std::endl;
            Crash();
        }

        throw T_Exception(fullMessage);
    }

    // Implementation for direct Throw<FatalError>("msg") etc.
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void ThrowWithLocation(std::source_location location, Ts const&... message)
    {
        auto prefix
            = concatenate(GetBaseFileName(location.file_name()), ":", location.line(), ": ");

        auto fullMessage = concatenate(prefix, message...);

        bool var = Error::BreakOnThrow();
        if(var)
        {
            std::cerr << fullMessage << std::endl;
            Crash();
        }

        throw T_Exception(fullMessage);
    }

    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(MessageWithLocation leadingMessage, Ts const&... messageParts)
    {
        ThrowWithLocation<T_Exception>(leadingMessage.loc, leadingMessage.message, messageParts...);
    }
}

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

#include <concepts>
#include <source_location>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>

#include <cassert>

#include <rocRoller/Utilities/Error_fwd.hpp>

namespace rocRoller
{
    struct Error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;

        template <typename Ta, typename Tb, typename... Ts>
        Error(Ta const&, Tb const&, Ts const&...);

        static bool BreakOnThrow();

        virtual const char* what() const noexcept override;

        void annotate(std::string const& msg);

    private:
        std::string m_annotatedMessage;
    };

    struct FatalError : public Error
    {
        using Error::Error;
    };

    struct RecoverableError : public Error
    {
        using Error::Error;
    };

    // Used by AssertError / AssertFatal / AssertRecoverable:
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void ThrowTagged(std::source_location location,
                                  const char*          exceptionTag,
                                  const char*          conditionText,
                                  Ts const&... message);

    // Used for plain Throw<FatalError>("msg", ...) calls:
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void ThrowWithLocation(std::source_location location, Ts const&... message);

    // Captures call–site location and forwards to ThrowWithLocation.
    template <typename T_Exception, typename... Ts>
    [[noreturn]] inline void Throw(Ts const&... message)
    {
        ThrowWithLocation<T_Exception>(std::source_location::current(), message...);
    }

    /**
     * Initiates a segfault.  This can be useful for debugging purposes.
     */
    [[noreturn]] void Crash();

    int* GetNullPointer();

    // Get path
    // Strips all "../" and "./"
    constexpr const char* GetBaseFileName(const char* file)
    {
        if(strnlen(file, 3) >= 3 && file[0] == '.' && file[1] == '.' && file[2] == '/')
        {
            return GetBaseFileName(file + 3);
        }
        else if(strnlen(file, 3) >= 2 && file[0] == '.' && file[1] == '/')
        {
            return GetBaseFileName(file + 2);
        }
        return file;
    }

#define ShowValue(var) concatenate("\t", #var, " = ", var, "\n")

#define AssertError(T_Exception, condition, message...)                                \
    do                                                                                 \
    {                                                                                  \
        bool condition_val = static_cast<bool>(condition);                             \
        if(!(condition_val))                                                           \
        {                                                                              \
            ThrowTagged<T_Exception>(                                                  \
                std::source_location::current(), #T_Exception, #condition, ##message); \
        }                                                                              \
    } while(0)

#define AssertFatal(...) AssertError(FatalError, __VA_ARGS__)
#define AssertRecoverable(...) AssertError(RecoverableError, __VA_ARGS__)
}

#include <rocRoller/Utilities/Error_impl.hpp>

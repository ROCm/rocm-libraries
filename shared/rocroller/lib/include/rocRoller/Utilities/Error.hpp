// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <source_location>
#include <stdexcept>
#include <string.h>
#include <string>
#include <string_view>
#include <utility>
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

    /**
     * MessageWithLocation is implicitly constructed from the first argument of
     * the direct Throw<>(...) overload. Because the constructor defaults the location to
     * std::source_location::current(), the captured location points at the user's call site.
    */

    struct MessageWithLocation
    {
        std::string          message;
        std::source_location loc;

        MessageWithLocation(std::string          msg,
                            std::source_location srcLoc = std::source_location::current())
            : message(std::move(msg))
            , loc(srcLoc)
        {
        }

        MessageWithLocation(std::string_view     msg,
                            std::source_location srcLoc = std::source_location::current())
            : message(msg)
            , loc(srcLoc)
        {
        }

        MessageWithLocation(const char*          msg,
                            std::source_location srcLoc = std::source_location::current())
            : message(msg ? msg : "")
            , loc(srcLoc)
        {
        }
    };

    /**
     * Throw overload used by AssertError / AssertFatal / AssertRecoverable.
     * Assert macros already capture source_location and tags explicitly.
     * Example:
     * // In user code:
     * AssertFatal(x < y, "oops");
     * // Conceptual expansion:
     * Throw<FatalError>(std::source_location::current(), "FatalError", "x < y", "oops");
     */
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(std::source_location location,
                            const char*          exceptionTag,
                            const char*          conditionText,
                            Ts const&... message);

    /**
     * Throw overload used for direct Throw<>(...) calls. The first message argument is 
     * wrapped in MessageWithLocation, which automatically captures source_location
     * at the call site and preserves it in the final error text.
     * Example:
     * // In user code:
     * Throw<FatalError>("prefix: ", ShowValue(x), "tail");
     * // Conceptual expansion:
     * Throw<FatalError>(MessageWithLocation("prefix: ", std::source_location::current()),
     *     ShowValue(x), "tail");
     */
    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(MessageWithLocation leadingMessage, Ts const&... messageParts);

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
            Throw<T_Exception>(                                                        \
                std::source_location::current(), #T_Exception, #condition, ##message); \
        }                                                                              \
    } while(0)

#define AssertFatal(...) AssertError(FatalError, __VA_ARGS__)
#define AssertRecoverable(...) AssertError(RecoverableError, __VA_ARGS__)
}

#include <rocRoller/Utilities/Error_impl.hpp>

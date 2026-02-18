// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(std::source_location location,
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

    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(MessageWithLocation leadingMessage, Ts const&... messageParts)
    {
        auto prefix = concatenate(
            GetBaseFileName(leadingMessage.loc.file_name()), ":", leadingMessage.loc.line(), ": ");

        auto fullMessage = concatenate(prefix, leadingMessage.message, messageParts...);

        bool var = Error::BreakOnThrow();
        if(var)
        {
            std::cerr << fullMessage << std::endl;
            Crash();
        }

        throw T_Exception(fullMessage);
    }
}

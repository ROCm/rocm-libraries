
#pragma once

#include "Error.hpp"
#include "Utils.hpp"

namespace rocRoller
{
    template <typename Ta, typename Tb, typename... Ts>
    Error::Error(Ta const& first, Tb const& second, Ts const&... vals)
        : Error(concatenate(first, second, vals...))
    {
    }

    template <typename T_Exception, typename... Ts>
    [[noreturn]] void Throw(Ts const&... message)
    {
        auto var = getenv(ENV_BREAK_ON_THROW.c_str());
        if(var && std::string(var) == "1")
        {
            std::cerr << concatenate(message...) << std::endl;
            Crash();
        }

        throw T_Exception(message...);
    }
}

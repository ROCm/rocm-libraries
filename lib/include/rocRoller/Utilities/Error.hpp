
#pragma once

#include <concepts>
#include <stdexcept>
#include <string>

#include <cassert>

#include "Error_fwd.hpp"

namespace rocRoller
{
    struct Error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;

        template <typename Ta, typename Tb, typename... Ts>
        Error(Ta const&, Tb const&, Ts const&...);
    };

    struct FatalError : public Error
    {
        using Error::Error;
    };

    struct RecoverableError : public Error
    {
        using Error::Error;
    };

    template <class T_Exception, typename... Ts>
    [[noreturn]] void Throw(Ts const&...);

    /**
     * Initiates a segfault.  This can be useful for debugging purposes.
     */
    [[noreturn]] void Crash();

    int* GetNullPointer();

    constexpr const char* GetBaseFileName(const char* file)
    {
        if(file[0] == '.' && file[1] == '.' && file[2] == '/')
            return file + 3;
        for(int i = 0; i < ROCROLLER_PATH_PREFIX_LENGTH + 1; ++i)
            assert(file[i] != '\0');
        return file + ROCROLLER_PATH_PREFIX_LENGTH + 1;
    }

#define ShowValue(var) concatenate("\t", #var, " = ", var, "\n")

#define AssertError(T_Exception, condition, message...)    \
    do                                                     \
    {                                                      \
        bool condition_val = static_cast<bool>(condition); \
        if(!(condition_val))                               \
            Throw<T_Exception>(GetBaseFileName(__FILE__),  \
                               ":",                        \
                               __LINE__,                   \
                               ": ",                       \
                               #T_Exception,               \
                               "(",                        \
                               #condition,                 \
                               ")\n",                      \
                               ##message);                 \
    } while(0)

#define AssertFatal(...) AssertError(FatalError, __VA_ARGS__)

#define AssertRecoverable(...) AssertError(RecoverableError, __VA_ARGS__)
}

#include "Error_impl.hpp"

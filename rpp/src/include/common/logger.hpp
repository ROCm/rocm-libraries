/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GUARD_RPP_LOGGER_HPP
#define GUARD_RPP_LOGGER_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <type_traits>

#include "each_args.hpp"
#include "object.hpp"

// See https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
#define RPP_PP_CAT(x, y) RPP_PP_PRIMITIVE_CAT(x, y)
#define RPP_PP_PRIMITIVE_CAT(x, y) x##y

#define RPP_PP_IIF(c) RPP_PP_PRIMITIVE_CAT(RPP_PP_IIF_, c)
#define RPP_PP_IIF_0(t, ...) __VA_ARGS__
#define RPP_PP_IIF_1(t, ...) t

#define RPP_PP_IS_PAREN(x) RPP_PP_IS_PAREN_CHECK(RPP_PP_IS_PAREN_PROBE x)
#define RPP_PP_IS_PAREN_CHECK(...) RPP_PP_IS_PAREN_CHECK_N(__VA_ARGS__, 0)
#define RPP_PP_IS_PAREN_PROBE(...) ~, 1,
#define RPP_PP_IS_PAREN_CHECK_N(x, n, ...) n

#define RPP_PP_EAT(...)
#define RPP_PP_EXPAND(...) __VA_ARGS__
#define RPP_PP_COMMA(...) ,

#define RPP_PP_TRANSFORM_ARGS(m, ...)                                 \
    RPP_PP_EXPAND(RPP_PP_PRIMITIVE_TRANSFORM_ARGS(m,               \
                                                        RPP_PP_COMMA, \
                                                        __VA_ARGS__,     \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        ()))

#define RPP_PP_EACH_ARGS(m, ...)                                    \
    RPP_PP_EXPAND(RPP_PP_PRIMITIVE_TRANSFORM_ARGS(m,             \
                                                        RPP_PP_EAT, \
                                                        __VA_ARGS__,   \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        ()))

#define RPP_PP_PRIMITIVE_TRANSFORM_ARGS(                                              \
    m, delim, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, ...) \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x0)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x1)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x1)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x2)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x2)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x3)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x3)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x4)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x4)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x5)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x5)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x6)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x6)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x7)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x7)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x8)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x8)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x9)                                         \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x9)                                             \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x10)                                        \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x10)                                            \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x11)                                        \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x11)                                            \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x12)                                        \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x12)                                            \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x13)                                        \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x13)                                            \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x14)                                        \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x14)                                            \
    RPP_PP_PRIMITIVE_TRANSFORM_ARG(delim, x15) RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x15)

#define RPP_PP_PRIMITIVE_TRANSFORM_ARG(m, x) \
    RPP_PP_IIF(RPP_PP_IS_PAREN(x))(RPP_PP_EAT, m)(x)

namespace rpp {

template <class Range>
std::ostream& LogRange(std::ostream& os, Range&& r, std::string delim)
{
    bool first = true;
    for(auto&& x : r)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << x;
    }
    return os;
}

template <class T, class... Ts>
std::array<T, sizeof...(Ts) + 1> make_array(T x, Ts... xs)
{
    return {{x, xs...}};
}

#define RPP_LOG_ENUM_EACH(x) std::pair<std::string, decltype(x)>(#x, x)
#ifdef _MSC_VER
#define RPP_LOG_ENUM(os, ...) os
#else
#define RPP_LOG_ENUM(os, x, ...) \
    rpp::LogEnum(os, x, make_array(RPP_PP_TRANSFORM_ARGS(RPP_LOG_ENUM_EACH, __VA_ARGS__)))
#endif

template <class T, class Range>
std::ostream& LogEnum(std::ostream& os, T x, Range&& values)
{
    auto it = std::find_if(values.begin(), values.end(), [&](auto&& p) { return p.second == x; });
    if(it == values.end())
        os << "Unknown: " << x;
    else
        os << it->first;
    return os;
}

enum class LoggingLevel
{
    Default = 0, // Warning level for Release builds, Info for Debug builds.
    Quiet   = 1, // None logging messages (except those controlled by RPP_ENABLE_LOGGING*).
    Fatal   = 2, // Fatal errors only (not used yet).
    Error   = 3, // Errors and fatals.
    Warning = 4, // All errors and warnings.
    Info    = 5, // All above plus information for debugging purposes.
    Info2   = 6, // All above  plus more detailed information for debugging.
    Trace   = 7  // The most detailed debugging messages
};

const char* LoggingLevelToCString(LoggingLevel level);
std::string LoggingPrefix();

/// \return true if level is enabled.
/// \param level - one of the values defined in LoggingLevel.
bool IsLogging(LoggingLevel level = LoggingLevel::Error);
bool IsLoggingCmd();
bool IsLoggingFunctionCalls();

template <class T>
auto LogObjImpl(T* x) -> decltype(get_object(*x))
{
    return get_object(*x);
}

inline void* LogObjImpl(void* x) { return x; }

inline const void* LogObjImpl(const void* x) { return x; }

#ifndef _MSC_VER
template <class T, typename std::enable_if<(std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << '\t' << name << " = ";
    if(x == nullptr)
        os << "nullptr";
    else
        os << LogObjImpl(x);
    return os;
}

template <class T, typename std::enable_if<(not std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << '\t' << name << " = " << get_object(x);
    return os;
}

#define RPP_LOG_FUNCTION_EACH(param)                                         \
    do                                                                          \
    {                                                                           \
        /* Clear temp stringstream & reset its state: */                        \
        std::ostringstream().swap(rpp_log_func_ss);                          \
        /* Use stringstram as ostream to engage existing template functions: */ \
        std::ostream& rpp_log_func_ostream = rpp_log_func_ss;             \
        rpp_log_func_ostream << rpp::LoggingPrefix();                     \
        rpp::LogParam(rpp_log_func_ostream, #param, param) << std::endl;  \
        std::cerr << rpp_log_func_ss.str();                                  \
    } while(false);

#define RPP_LOG_FUNCTION(...)                                                        \
    do                                                                                  \
        if(rpp::IsLoggingFunctionCalls())                                            \
        {                                                                               \
            std::ostringstream rpp_log_func_ss;                                      \
            rpp_log_func_ss << rpp::LoggingPrefix() << __PRETTY_FUNCTION__ << "{" \
                               << std::endl;                                            \
            std::cerr << rpp_log_func_ss.str();                                      \
            RPP_PP_EACH_ARGS(RPP_LOG_FUNCTION_EACH, __VA_ARGS__)                  \
            std::ostringstream().swap(rpp_log_func_ss);                              \
            rpp_log_func_ss << rpp::LoggingPrefix() << "}" << std::endl;          \
            std::cerr << rpp_log_func_ss.str();                                      \
        }                                                                               \
    while(false)
#else
#define RPP_LOG_FUNCTION(...)
#endif

std::string LoggingParseFunction(const char* func, const char* pretty_func);

#define RPP_LOG(level, ...)                                                               \
    // do                                                                                       \
    // {                                                                                        \
    //     if(rpp::IsLogging(level))                                                         \
    //     {                                                                                    \
    //         std::ostringstream rpp_log_ss;                                                \
    //         rpp_log_ss << rpp::LoggingPrefix() << LoggingLevelToCString(level) << " [" \
    //                       << rpp::LoggingParseFunction(__func__,            /* NOLINT */  \
    //                                                       __PRETTY_FUNCTION__) /* NOLINT */  \
    //                       << "] " << __VA_ARGS__ << std::endl;                               \
    //         std::cerr << rpp_log_ss.str();                                                \
    //     }                                                                                    \
    // } while(false)

#define RPP_LOG_E(...) RPP_LOG(rpp::LoggingLevel::Error, __VA_ARGS__)
#define RPP_LOG_W(...) RPP_LOG(rpp::LoggingLevel::Warning, __VA_ARGS__)
#define RPP_LOG_I(...) RPP_LOG(rpp::LoggingLevel::Info, __VA_ARGS__)
#define RPP_LOG_I2(...) RPP_LOG(rpp::LoggingLevel::Info2, __VA_ARGS__)
#define RPP_LOG_T(...) RPP_LOG(rpp::LoggingLevel::Trace, __VA_ARGS__)

#define RPP_LOG_DRIVER_CMD(...)                                                      \
    do                                                                                  \
    {                                                                                   \
        std::ostringstream rpp_driver_cmd_ss;                                        \
        rpp_driver_cmd_ss << rpp::LoggingPrefix() << "Command"                    \
                             << " [" << rpp::LoggingParseFunction(                   \
                                            __func__, __PRETTY_FUNCTION__) /* NOLINT */ \
                             << "] ./bin/MIOpenDriver " << __VA_ARGS__ << std::endl;    \
        std::cerr << rpp_driver_cmd_ss.str();                                        \
    } while(false)

} // namespace rpp

#endif

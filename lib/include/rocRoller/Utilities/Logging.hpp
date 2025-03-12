/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 Advanced Micro Devices, Inc.
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

// Only need the forward declaration of LogLevel
#include <rocRoller/Utilities/Settings_fwd.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <string>

namespace rocRoller
{
    namespace Log
    {

        class logger
        {
        public:
            void log(LogLevel level, const std::string& str);

            template <typename... Args>
            void log(LogLevel level, fmt::format_string<Args...> formatStr, Args&&... args)
            {
                if(should_log(level))
                {
                    std::string buf;
                    fmt::vformat_to(
                        std::back_inserter(buf), formatStr, fmt::make_format_args(args...));
                    log(level, buf);
                }
            }

#define add_logging_method(method_name, level_name)                                \
    inline void method_name(const std::string& str)                                \
    {                                                                              \
        log(LogLevel::level_name, str);                                            \
    }                                                                              \
    template <typename... Args>                                                    \
    inline void method_name(fmt::format_string<Args...> formatStr, Args&&... args) \
    {                                                                              \
        log(LogLevel::level_name, formatStr, std::forward<Args>(args)...);         \
    }

            add_logging_method(trace, Trace);
            add_logging_method(debug, Debug);
            add_logging_method(info, Info);
            add_logging_method(warn, Warning);
            add_logging_method(error, Error);
            add_logging_method(critical, Critical);

            bool should_log(LogLevel msg_level) const;
        };

        using LoggerPtr = std::shared_ptr<logger>;
        LoggerPtr getLogger();

        inline void log(LogLevel level, const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->log(level, str);
        }

        template <typename... Args>
        inline void log(LogLevel level, fmt::format_string<Args...> formatStr, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->log(level, formatStr, std::forward<Args>(args)...);
        }

#define declare_logging_function(name)                                      \
    inline void name(const std::string& str)                                \
    {                                                                       \
        static auto defaultLog = getLogger();                               \
        defaultLog->name(str);                                              \
    }                                                                       \
    template <typename... Args>                                             \
    inline void name(fmt::format_string<Args...> formatStr, Args&&... args) \
    {                                                                       \
        static auto defaultLog = getLogger();                               \
        defaultLog->name(formatStr, std::forward<Args>(args)...);           \
    }

        declare_logging_function(trace);
        declare_logging_function(debug);
        declare_logging_function(info);
        declare_logging_function(warn);
        declare_logging_function(error);
        declare_logging_function(critical);

    } // namespace Log
} // namespace rocRoller

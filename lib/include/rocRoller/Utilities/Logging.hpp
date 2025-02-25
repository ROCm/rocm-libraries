/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2024 Advanced Micro Devices, Inc.
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

#include <string>

// #define SPDLOG_COMPILED_LIB 1
#include <spdlog/spdlog.h>

// In order for linking to work for GCC, any includes from fmt must happen after including spdlog.h
#include <spdlog/fmt/ranges.h>

namespace rocRoller
{
    namespace Log
    {
        using LoggerPtr = std::shared_ptr<spdlog::logger>;
        LoggerPtr getLogger();

        inline void trace(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->trace(str);
        }

        template <typename... Args>
        inline void trace(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->trace(fmt, std::forward<Args>(args)...);
        }

        inline void debug(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->debug(str);
        }

        template <typename... Args>
        inline void debug(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->debug(fmt, std::forward<Args>(args)...);
        }

        inline void info(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->info(str);
        }

        template <typename... Args>
        inline void info(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->info(fmt, std::forward<Args>(args)...);
        }

        inline void warn(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->warn(str);
        }

        template <typename... Args>
        inline void warn(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->warn(fmt, std::forward<Args>(args)...);
        }

        inline void error(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->error(str);
        }

        template <typename... Args>
        inline void error(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->error(fmt, std::forward<Args>(args)...);
        }

        inline void critical(const std::string& str)
        {
            static auto defaultLog = getLogger();
            defaultLog->critical(str);
        }

        template <typename... Args>
        inline void critical(spdlog::format_string_t<Args...> fmt, Args&&... args)
        {
            static auto defaultLog = getLogger();
            defaultLog->critical(fmt, std::forward<Args>(args)...);
        }
    } // namespace Log
} // namespace rocRoller

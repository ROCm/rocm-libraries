/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocsparse_getenv.hpp provides an OS-agnostic way to read environment
 *  variables. Windows uses the WinAPI, Linux uses the POSIX API. No other
 *  operating system is supported.
 */

#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <cstdlib>
#endif

#include <optional>
#include <string>
#include <string_view>

namespace rocsparse
{
    ///
    /// @brief Return the value of an environment variable, if defined.
    /// @param name Name of the environment variable to look up.
    /// @return The value of the environment variable, or std::nullopt if it is not defined.
    ///
    inline std::optional<std::string> get_environment_variable(std::string_view name)
    {
        // WinAPI and POSIX both require a null-terminated C string.
        const std::string name_str(name);
#if defined(_WIN32)
        DWORD required_size = GetEnvironmentVariableA(name_str.c_str(), nullptr, 0);
        if(required_size == 0)
        {
            return std::nullopt;
        }
        std::string value(required_size - 1, '\0');
        GetEnvironmentVariableA(name_str.c_str(), value.data(), required_size);
        return value;
#elif defined(__linux__)
        const char* value = std::getenv(name_str.c_str());
        if(value == nullptr)
        {
            return std::nullopt;
        }
        return std::string(value);
#endif
    }
}

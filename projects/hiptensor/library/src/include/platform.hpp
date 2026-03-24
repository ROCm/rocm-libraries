/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <hiptensor/hiptensor.h>

// Cross-platform safe file open (definition in platform.cpp; not exported from DLL)
FILE* safeFopen(const char* filename, const char* mode);

// Cross-platform safe environment variable access
inline std::optional<std::string> getEnvironmentVariable(const char* name)
{
#ifdef _WIN32
    char   buffer[256];
    size_t required_size = 0;
    if(getenv_s(&required_size, buffer, sizeof(buffer), name) != 0
       || required_size == 0 || required_size > sizeof(buffer))
    {
        return std::nullopt;
    }
    return std::string(buffer);
#else
    const char* val = std::getenv(name);
    return val ? std::optional<std::string>{val} : std::nullopt;
#endif
}

// Cross-platform safe localtime: fills buff with a formatted timestamp, or "TIMESTAMP-ERROR" on failure
void safeLocaltime(char* buff, size_t buffSize, const time_t* t);

// Cross-platform process ID
inline int getProcessId()
{
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

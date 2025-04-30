// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// wrappers around environment variable routines

#pragma once

#include <string>

// Windows provides "getenv" and "_putenv", but those modify the
// runtime's copy of the environment.  The actual environment in the
// process control block is accessed using GetEnvironmentVariable and
// SetEnvironmentVariable.

#ifdef WIN32
#include <windows.h>
static void rocfft_setenv(const char* var, const char* value)
{
    SetEnvironmentVariable(var, value);
}
static void rocfft_unsetenv(const char* var)
{
    SetEnvironmentVariable(var, nullptr);
}
static std::string rocfft_getenv(const char* var)
{
    DWORD       size = GetEnvironmentVariable(var, nullptr, 0);
    std::string ret;
    if(size)
    {
        ret.resize(size);
        GetEnvironmentVariable(var, ret.data(), size);
        // GetEnvironmentVariable counts the terminating null, so remove it
        while(!ret.empty() && ret.back() == 0)
            ret.pop_back();
    }
    return ret;
}

#else

#include <stdlib.h>

static void rocfft_setenv(const char* var, const char* value)
{
    setenv(var, value, 1);
}
static void rocfft_unsetenv(const char* var)
{
    unsetenv(var);
}
static std::string rocfft_getenv(const char* var)
{
    auto value = getenv(var);
    return value ? value : "";
}
#endif

// RAII object to set an environment variable and restore it to its
// previous value on destruction
struct EnvironmentSetTemp
{
    EnvironmentSetTemp(const char* _var, const char* val)
        : var(_var)
    {
        auto val_ptr = rocfft_getenv(_var);
        if(!val_ptr.empty())
            oldvalue = val_ptr;
        rocfft_setenv(_var, val);
    }
    ~EnvironmentSetTemp()
    {
        if(oldvalue.empty())
            rocfft_unsetenv(var.c_str());
        else
            rocfft_setenv(var.c_str(), oldvalue.c_str());
    }
    std::string var;
    std::string oldvalue;
};

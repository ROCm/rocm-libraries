// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_LIBRARY_PATH
#define ROCFFT_LIBRARY_PATH

#include "rocfft/rocfft.h"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

#ifdef WIN32
static std::filesystem::path get_library_path_win32()
{
    static const char* ROCFFT_DLL = "rocfft.dll";
    // get module handle for rocfft lib
    HMODULE module = GetModuleHandleA(ROCFFT_DLL);
    if(!module)
        throw std::runtime_error("unable to find dll handle");

    char library_path[MAX_PATH];
    if(GetModuleFileNameA(module, library_path, MAX_PATH) == MAX_PATH)
        throw std::runtime_error("unable to get path to dll");

    return library_path;
}
#else
static std::filesystem::path get_library_path_unix()
{
    // get address of rocfft lib by looking for a symbol in it
    Dl_info   info;
    link_map* map = nullptr;
    if(!dladdr1(reinterpret_cast<const void*>(rocfft_plan_create),
                &info,
                reinterpret_cast<void**>(&map),
                RTLD_DL_LINKMAP))
        throw std::runtime_error("dladdr failed");
    return map->l_name;
}
#endif

static std::filesystem::path get_library_path()
{
    // library is not a separate thing at runtime, if we're building
    // static
#ifdef ROCFFT_STATIC_LIB
    return {};
#else
#ifdef WIN32
    return get_library_path_win32();
#else
    return get_library_path_unix();
#endif
#endif
}

#endif

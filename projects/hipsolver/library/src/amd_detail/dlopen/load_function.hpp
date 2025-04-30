/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "lib_macros.hpp"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif /* _WIN32 */

HIPSOLVER_BEGIN_NAMESPACE

template <typename Fn>
bool load_function(void* handle, const char* symbol, Fn& fn)
{
#ifndef HIPSOLVER_STATIC_LIB
#ifdef _WIN32
    fn       = (Fn)(GetProcAddress((HMODULE)handle, symbol));
    bool err = !fn;
#else
    fn        = (Fn)(dlsym(handle, symbol));
    char* err = dlerror(); // clear errors
#ifndef NDEBUG
    if(err)
        std::cerr << "hipsolver: error loading " << symbol << ": " << err << std::endl;
#endif
#endif /* _WIN32 */
    return !err;
#else /* HIPSOLVER_STATIC_LIB */
    return false;
#endif
}

HIPSOLVER_END_NAMESPACE

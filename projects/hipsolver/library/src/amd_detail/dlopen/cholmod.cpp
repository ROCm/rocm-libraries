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

#include "cholmod.hpp"
#include "load_function.hpp"

HIPSOLVER_BEGIN_NAMESPACE

fp_cholmod_start            g_cholmod_start;
fp_cholmod_finish           g_cholmod_finish;
fp_cholmod_allocate_sparse  g_cholmod_allocate_sparse;
fp_cholmod_free_sparse      g_cholmod_free_sparse;
fp_cholmod_allocate_dense   g_cholmod_allocate_dense;
fp_cholmod_free_dense       g_cholmod_free_dense;
fp_cholmod_free_factor      g_cholmod_free_factor;
fp_cholmod_drop             g_cholmod_drop;
fp_cholmod_analyze          g_cholmod_analyze;
fp_cholmod_analyze_ordering g_cholmod_analyze_ordering;
fp_cholmod_factorize        g_cholmod_factorize;
fp_cholmod_solve            g_cholmod_solve;

static bool load_cholmod()
{
#ifndef HIPSOLVER_STATIC_LIB
#ifdef _WIN32
    // Library users will need to call SetErrorMode(SEM_FAILCRITICALERRORS) if
    // they wish to avoid an error message box when this library is not found.
    // The call is not done by hipSOLVER directly, as it is not thread-safe and
    // will affect the global state of the program.
    void* handle = LoadLibraryW(L"cholmod.dll");
#else
    void* handle = dlopen("libcholmod.so.5", RTLD_NOW | RTLD_LOCAL);
    char* err    = dlerror(); // clear errors

    if(!handle)
    {
        handle = dlopen("libcholmod.so.4", RTLD_NOW | RTLD_LOCAL);
        err    = dlerror(); // clear errors
    }

    if(!handle)
    {
        handle = dlopen("libcholmod.so.3", RTLD_NOW | RTLD_LOCAL);
        err    = dlerror(); // clear errors
    }

    if(!handle)
    {
        handle = dlopen("libcholmod.so", RTLD_NOW | RTLD_LOCAL);
        err    = dlerror(); // clear errors
    }
#ifndef NDEBUG
    if(!handle)
        std::cerr << "hipsolver: error loading libcholmod.so: " << err << std::endl;
#endif
#endif /* _WIN32 */
    if(!handle)
        return false;

    if(!load_function(handle, "cholmod_start", g_cholmod_start))
        return false;
    if(!load_function(handle, "cholmod_finish", g_cholmod_finish))
        return false;

    if(!load_function(handle, "cholmod_allocate_sparse", g_cholmod_allocate_sparse))
        return false;
    if(!load_function(handle, "cholmod_free_sparse", g_cholmod_free_sparse))
        return false;
    if(!load_function(handle, "cholmod_allocate_dense", g_cholmod_allocate_dense))
        return false;
    if(!load_function(handle, "cholmod_free_dense", g_cholmod_free_dense))
        return false;
    if(!load_function(handle, "cholmod_free_factor", g_cholmod_free_factor))
        return false;

    if(!load_function(handle, "cholmod_drop", g_cholmod_drop))
        return false;
    if(!load_function(handle, "cholmod_analyze", g_cholmod_analyze))
        return false;
    if(!load_function(handle, "cholmod_analyze_ordering", g_cholmod_analyze_ordering))
        return false;
    if(!load_function(handle, "cholmod_factorize", g_cholmod_factorize))
        return false;
    if(!load_function(handle, "cholmod_solve", g_cholmod_solve))
        return false;

    return true;
#else /* HIPSOLVER_STATIC_LIB */
    return false;
#endif
}

bool try_load_cholmod()
{
    // Function-scope static initialization has been thread-safe since C++11.
    // There is an implicit mutex guarding the initialization.
    static bool result = load_cholmod();
    return result;
}

HIPSOLVER_END_NAMESPACE

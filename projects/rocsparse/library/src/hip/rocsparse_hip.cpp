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
#include "rocsparse_hip.hpp"
#include "rocsparse_hip_debug_t.hpp"
#include <iostream>
#ifdef ROCSPARSE_WITH_MEMSTAT
#include "rocsparse-auxiliary.h"
#include "rocsparse_memstat.hpp"

namespace rocsparse
{
    namespace hip
    {
        static std::string
            source_tag(const char* function_name, const char* file_name, int32_t file_line)
        {
            std::stringstream s;
            s << "from " << function_name << " at line " << file_line << " in " << file_name;
            return s.str();
        }
    }
}
#define ROCSPARSE_HIP_SOURCE_TAG(FCT, F, L) rocsparse::hip::source_tag(FCT, F, L).c_str()
// #define ROCSPARSE_HIP_SOURCE_MSG(msg_) #msg_
//#define ROCSPARSE_HIP_SOURCE_TAG(FCT,F,L) __FILE__ " " ROCSPARSE_HIP_SOURCE_MSG(msg_)
#else
#endif

hipError_t rocsparse::hip::hipMemcpy2DAsync(void*         target,
                                            size_t        tpitch,
                                            const void*   source,
                                            size_t        spitch,
                                            size_t        width,
                                            size_t        height,
                                            hipMemcpyKind kind,
                                            hipStream_t   stream,
                                            const char*   function_name,
                                            const char*   file_name,
                                            int32_t       file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipMemcpy2DAsync(target, tpitch, source, spitch, width, height, kind, stream);
    }
    else
    {
        return rocsparse::hip::debug_t::hipMemcpy2DAsync(target,
                                                         tpitch,
                                                         source,
                                                         spitch,
                                                         width,
                                                         height,
                                                         kind,
                                                         stream,
                                                         {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipDeviceSynchronize(const char* function_name,
                                                const char* file_name,
                                                int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipDeviceSynchronize();
    }
    else
    {
        return rocsparse::hip::debug_t::hipDeviceSynchronize({function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipStreamSynchronize(hipStream_t stream,
                                                const char* function_name,
                                                const char* file_name,
                                                int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipStreamSynchronize(stream);
    }
    else
    {
        return rocsparse::hip::debug_t::hipStreamSynchronize(stream,
                                                             {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMemsetAsync(void*       target,
                                          int         value,
                                          size_t      size_in_bytes,
                                          hipStream_t stream,
                                          const char* function_name,
                                          const char* file_name,
                                          int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipMemsetAsync(target, value, size_in_bytes, stream);
    }
    else
    {
        return rocsparse::hip::debug_t::hipMemsetAsync(
            target, value, size_in_bytes, stream, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMemset(void*       target,
                                     int         value,
                                     size_t      size_in_bytes,
                                     const char* function_name,
                                     const char* file_name,
                                     int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipMemset(target, value, size_in_bytes);
    }
    else
    {
        return rocsparse::hip::debug_t::hipMemset(
            target, value, size_in_bytes, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMemcpy(void*         target,
                                     const void*   source,
                                     size_t        size_in_bytes,
                                     hipMemcpyKind kind,
                                     const char*   function_name,
                                     const char*   file_name,
                                     int32_t       file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipMemcpy(target, source, size_in_bytes, kind);
    }
    else
    {
        return rocsparse::hip::debug_t::hipMemcpy(
            target, source, size_in_bytes, kind, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMemcpyAsync(void*         target,
                                          const void*   source,
                                          size_t        size_in_bytes,
                                          hipMemcpyKind kind,
                                          hipStream_t   stream,
                                          const char*   function_name,
                                          const char*   file_name,
                                          int32_t       file_line)
{

    if(!rocsparse::hip::debug_t::enabled())
    {
        return ::hipMemcpyAsync(target, source, size_in_bytes, kind, stream);
    }
    else
    {
        return rocsparse::hip::debug_t::hipMemcpyAsync(
            target, source, size_in_bytes, kind, stream, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMalloc(void**      p,
                                     size_t      size_in_bytes,
                                     const char* function_name,
                                     const char* file_name,
                                     int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
#ifndef ROCSPARSE_WITH_MEMSTAT
        return ::hipMalloc(p, size_in_bytes);
#else
        return rocsparse_hip_malloc(
            p, size_in_bytes, ROCSPARSE_HIP_SOURCE_TAG(function_name, file_name, file_line));
#endif
    }
    else
    {
        return rocsparse::hip::debug_t::hipMalloc(
            p, size_in_bytes, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipMallocAsync(void**      p,
                                          size_t      size_in_bytes,
                                          hipStream_t stream,
                                          const char* function_name,
                                          const char* file_name,
                                          int32_t     file_line)
{
#if HIP_VERSION >= 50300000
    if(!rocsparse::hip::debug_t::enabled())
    {
#ifndef ROCSPARSE_WITH_MEMSTAT
        return ::hipMallocAsync(p, size_in_bytes, stream);
#else
        return rocsparse_hip_malloc_async(
            p,
            size_in_bytes,
            stream,
            ROCSPARSE_HIP_SOURCE_TAG(function_name, file_name, file_line));
#endif
    }
    else
    {
        return rocsparse::hip::debug_t::hipMallocAsync(
            p, size_in_bytes, stream, {function_name, file_name, file_line});
    }
#else
    return rocsparse::hip::hipMalloc(p, size_in_bytes, {function_name, file_name, file_line});
#endif
}

hipError_t rocsparse::hip::hipFree(void*       p,
                                   const char* function_name,
                                   const char* file_name,
                                   int32_t     file_line)
{
    if(!rocsparse::hip::debug_t::enabled())
    {
#ifndef ROCSPARSE_WITH_MEMSTAT
        return ::hipFree(p);
#else
        return rocsparse_hip_free(p, ROCSPARSE_HIP_SOURCE_TAG(function_name, file_name, file_line));
#endif
    }
    else
    {
        return rocsparse::hip::debug_t::hipFree(p, {function_name, file_name, file_line});
    }
}

hipError_t rocsparse::hip::hipFreeAsync(void*       p,
                                        hipStream_t stream,
                                        const char* function_name,
                                        const char* file_name,
                                        int32_t     file_line)
{
    if(p == nullptr)
        return hipSuccess;

#if HIP_VERSION >= 50300000
    if(!rocsparse::hip::debug_t::enabled())
    {
#ifndef ROCSPARSE_WITH_MEMSTAT
        return ::hipFreeAsync(p, stream);
#else
        return rocsparse_hip_free_async(
            p, stream, ROCSPARSE_HIP_SOURCE_TAG(function_name, file_name, file_line));
#endif
    }
    else
    {
        return rocsparse::hip::debug_t::hipFreeAsync(
            p, stream, {function_name, file_name, file_line});
    }
#else
    return rocsparse::hip::hipFree(p);
#endif
}

void rocsparse::hip::tag_hipLaunchKernelGGL(hipStream_t stream,
                                            const char* function_name,
                                            const char* file_name,
                                            int32_t     file_line)
{
    if(rocsparse::hip::debug_t::enabled())
    {
        rocsparse::hip::debug_t::tag_hipLaunchKernelGGL(stream,
                                                        {function_name, file_name, file_line});
    }
}

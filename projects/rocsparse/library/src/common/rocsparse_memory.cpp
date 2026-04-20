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
#include "rocsparse_memory.hpp"

hipError_t rocsparse_hipMemcpy2DAsync(void*         target,
                                      size_t        tpitch,
                                      const void*   source,
                                      size_t        spitch,
                                      size_t        width,
                                      size_t        height,
                                      hipMemcpyKind kind,
                                      hipStream_t   stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemcpy2DAsync(target, tpitch, source, spitch, width, height, kind, stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_memcpy2D_async(
            target, tpitch, source, spitch, width, height, kind, stream);
    }
}

hipError_t rocsparse_hipDeviceSynchronize()
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipDeviceSynchronize();
    }
    else
    {
        static constexpr hipStream_t default_stream{};
        auto&                        info = rocsparse::memory_debug_t::get_info(default_stream);
        return info.call_device_synchronize();
    }
}

hipError_t rocsparse_hipStreamSynchronize(hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipStreamSynchronize(stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_stream_synchronize(stream);
    }
}

hipError_t
    rocsparse_hipMemsetAsync(void* target, int value, size_t size_in_bytes, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemsetAsync(target, value, size_in_bytes, stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_memset_async(target, value, size_in_bytes, stream);
    }
}

hipError_t rocsparse_hipMemset(void* target, int value, size_t size_in_bytes)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemset(target, value, size_in_bytes);
    }
    else
    {
        static constexpr hipStream_t default_stream{};
        auto&                        info = rocsparse::memory_debug_t::get_info(default_stream);
        return info.call_memset(target, value, size_in_bytes);
    }
}

hipError_t
    rocsparse_hipMemcpy(void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemcpy(target, source, size_in_bytes, kind);
    }
    else
    {
        static constexpr hipStream_t default_stream{};
        auto&                        info = rocsparse::memory_debug_t::get_info(default_stream);
        return info.call_memcpy(target, source, size_in_bytes, kind);
    }
}

hipError_t rocsparse_hipMemcpyAsync(
    void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemcpyAsync(target, source, size_in_bytes, kind, stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_memcpy_async(target, source, size_in_bytes, kind, stream);
    }
}

hipError_t rocsparse_hipMalloc_impl(void** p, size_t size_in_bytes)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMalloc(p, size_in_bytes);
    }
    else
    {
        static constexpr hipStream_t default_stream{};
        auto&                        info = rocsparse::memory_debug_t::get_info(default_stream);
        return info.call_malloc(p, size_in_bytes);
    }
}

hipError_t rocsparse_hipMallocAsync_impl(void** p, size_t size_in_bytes, hipStream_t stream)
{
#if HIP_VERSION >= 50300000
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMallocAsync(p, size_in_bytes, stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_malloc_async(p, size_in_bytes, stream);
    }
#else
    return rocsparse_hipMalloc(p_, size_);
#endif
}

hipError_t rocsparse_hipFree(void* p)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipFree(p);
    }
    else
    {
        static constexpr hipStream_t default_stream{};
        auto&                        info = rocsparse::memory_debug_t::get_info(default_stream);
        return info.call_free(p);
    }
}

hipError_t rocsparse_hipFreeAsync(void* p, hipStream_t stream)
{
#if HIP_VERSION >= 50300000
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipFreeAsync(p, stream);
    }
    else
    {
        auto& info = rocsparse::memory_debug_t::get_info(stream);
        return info.call_free_async(p, stream);
    }
#else
    return rocsparse_hipFree(p);
#endif
}

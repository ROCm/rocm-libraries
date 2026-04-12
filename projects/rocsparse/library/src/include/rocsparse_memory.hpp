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

#pragma once

#include "rocsparse_debug.hpp"
#include <atomic>
#include <hip/hip_runtime_api.h>
namespace rocsparse
{

    struct execution_t
    {
    public:
        typedef enum func_t_
        {
            hip_malloc,
            hip_free,
            hip_malloc_async,
            hip_free_async,
            hip_memcpy,
            hip_memcpy_async,
            hip_memset,
            hip_memset_async,
            hip_stream_synchronize,
            hip_device_synchronize,
            hip_launch_kernel
        } func_t;

        static constexpr int32_t func_size = 11;

    private:
        std::atomic<int64_t> stack_count{0};
        std::atomic<int64_t> count_calls[func_size]{};
        std::atomic<int64_t> ncalls{0};
        execution_t() = default;
        std::atomic<func_t> m_last_call{(func_t)-1};

    public:
        ~execution_t();
        int64_t             get_ncalls() const;
        func_t              get_last_call() const;
        void                set_last_call(func_t);
        void                flag_kernel_launch();
        static execution_t& instance();

        bool    is_memory_stack_clean() const;
        bool    hit_stream_synchronize() const;
        bool    hit_device_synchronize() const;
        bool    hit_synchronize() const;
        void    reset();
        void    info() const;
        int64_t count(func_t func) const;

        hipError_t
            call_memcpy(void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind);
        hipError_t call_memcpy_async(void*         target,
                                     const void*   source,
                                     size_t        size_in_bytes,
                                     hipMemcpyKind kind,
                                     hipStream_t   stream);
        hipError_t call_memset(void* target, int value, size_t size_in_bytes);
        hipError_t
            call_memset_async(void* target, int value, size_t size_in_bytes, hipStream_t stream);
        hipError_t call_device_synchronize();
        hipError_t call_stream_synchronize(hipStream_t stream);
        hipError_t call_malloc_async(void** p_that, size_t size_in_bytes, hipStream_t stream);
        hipError_t call_free_async(void* that, hipStream_t stream);
        hipError_t call_malloc(void** p_that, size_t size_in_bytes);
        hipError_t call_free(void* that);
    };

}

inline hipError_t rocsparse_hipDeviceSynchronize()
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipDeviceSynchronize();
    }
    else
    {
        return rocsparse::execution_t::instance().call_device_synchronize();
    }
}

inline hipError_t rocsparse_hipStreamSynchronize(hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipStreamSynchronize(stream);
    }
    else
    {
        return rocsparse::execution_t::instance().call_stream_synchronize(stream);
    }
}

inline hipError_t
    rocsparse_hipMemsetAsync(void* target, int value, size_t size_in_bytes, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemsetAsync(target, value, size_in_bytes, stream);
    }
    else
    {
        return rocsparse::execution_t::instance().call_memset_async(
            target, value, size_in_bytes, stream);
    }
}

inline hipError_t rocsparse_hipMemset(void* target, int value, size_t size_in_bytes)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemset(target, value, size_in_bytes);
    }
    else
    {
        return rocsparse::execution_t::instance().call_memset(target, value, size_in_bytes);
    }
}

inline hipError_t
    rocsparse_hipMemcpy(void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemcpy(target, source, size_in_bytes, kind);
    }
    else
    {
        return rocsparse::execution_t::instance().call_memcpy(target, source, size_in_bytes, kind);
    }
}

inline hipError_t rocsparse_hipMemcpyAsync(
    void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMemcpyAsync(target, source, size_in_bytes, kind, stream);
    }
    else
    {
        return rocsparse::execution_t::instance().call_memcpy_async(
            target, source, size_in_bytes, kind, stream);
    }
}

inline hipError_t rocsparse_hipMalloc_impl(void** p, size_t size_in_bytes)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMalloc(p, size_in_bytes);
    }
    else
    {
        return rocsparse::execution_t::instance().call_malloc(p, size_in_bytes);
    }
}

#define rocsparse_hipMalloc(p_, size_) \
    rocsparse_hipMalloc_impl(reinterpret_cast<void**>((p_)), (size_))

inline hipError_t rocsparse_hipMallocAsync_impl(void** p, size_t size_in_bytes, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipMallocAsync(p, size_in_bytes, stream);
    }
    else
    {
        return rocsparse::execution_t::instance().call_malloc_async(p, size_in_bytes, stream);
    }
}

#define rocsparse_hipMallocAsync(p_, size_, stream_) \
    rocsparse_hipMallocAsync_impl(reinterpret_cast<void**>((p_)), (size_), stream_)

inline hipError_t rocsparse_hipFree(void* p)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipFree(p);
    }
    else
    {
        return rocsparse::execution_t::instance().call_free(p);
    }
}

inline hipError_t rocsparse_hipFreeAsync(void* p, hipStream_t stream)
{
    if(false == rocsparse_debug_variables.get_debug())
    {
        return hipFreeAsync(p, stream);
    }
    else
    {
        return rocsparse::execution_t::instance().call_free_async(p, stream);
    }
}

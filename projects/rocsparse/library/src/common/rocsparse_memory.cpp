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
#include "rocsparse-export.h"
#include "rocsparse-types.h"

int64_t rocsparse::execution_t::get_ncalls() const
{
    return this->ncalls;
}

int64_t rocsparse::execution_t::count(func_t func) const
{
    return this->count_calls[func];
}

bool rocsparse::execution_t::is_memory_stack_clean() const
{
    return (this->stack_count == 0);
}

bool rocsparse::execution_t::hit_stream_synchronize() const
{
    return (this->count_calls[func_t::hip_stream_synchronize] > 0);
}

bool rocsparse::execution_t::hit_device_synchronize() const
{
    return (this->count_calls[func_t::hip_device_synchronize] > 0);
}

bool rocsparse::execution_t::hit_synchronize() const
{
    return this->hit_stream_synchronize() || this->hit_device_synchronize();
}

void rocsparse::execution_t::reset()
{
    this->stack_count = 0;
    this->ncalls      = 0;
    for(int32_t i = 0; i < func_size; ++i)
    {
        this->count_calls[i] = 0;
    }
}

extern "C" {
ROCSPARSE_EXPORT void    rocsparse_execution_reset(rocsparse_handle handle);
ROCSPARSE_EXPORT int32_t rocsparse_execution_is_stream_synchronized(rocsparse_handle handle);

ROCSPARSE_EXPORT int32_t rocsparse_execution_is_asynchronous(rocsparse_handle handle);
ROCSPARSE_EXPORT int32_t rocsparse_execution_is_synchronous(rocsparse_handle handle);
ROCSPARSE_EXPORT int32_t rocsparse_execution_is_partially_synchronous(rocsparse_handle handle);
ROCSPARSE_EXPORT int32_t rocsparse_execution_is_host(rocsparse_handle handle);
}

namespace rocsparse
{
    using exec_t      = rocsparse::execution_t;
    static auto* exec = &rocsparse::execution_t::instance();
}

extern "C" int32_t rocsparse_execution_is_partially_synchronous(rocsparse_handle handle)
{
    const auto last_call = rocsparse::exec->get_last_call();
    const auto ncalls    = rocsparse::exec->get_ncalls();
    if(ncalls == 0)
        return 0;
    return (((rocsparse::exec->count(rocsparse::execution_t::hip_memcpy) > 0)
             || (rocsparse::exec->count(rocsparse::execution_t::hip_malloc) > 0)
             || (rocsparse::exec->count(rocsparse::execution_t::hip_free) > 0)
             || (rocsparse::exec->count(rocsparse::execution_t::hip_memset) > 0)
             || (rocsparse::exec->count(rocsparse::execution_t::hip_stream_synchronize) > 0)
             || (rocsparse::exec->count(rocsparse::execution_t::hip_device_synchronize) > 0))
            && ((last_call != rocsparse::execution_t::hip_memcpy)
                && (last_call != rocsparse::execution_t::hip_malloc)
                && (last_call != rocsparse::execution_t::hip_free)
                && (last_call != rocsparse::execution_t::hip_memset)
                && (last_call != rocsparse::execution_t::hip_stream_synchronize)
                && (last_call != rocsparse::execution_t::hip_device_synchronize)))
               ? 1
               : 0;
}

extern "C" int32_t rocsparse_execution_is_asynchronous(rocsparse_handle handle)
{
    const auto ncalls = rocsparse::exec->get_ncalls();
    if(ncalls == 0)
        return 0;
    return (((rocsparse::exec->count(rocsparse::execution_t::hip_memcpy) == 0)
             && (rocsparse::exec->count(rocsparse::execution_t::hip_malloc) == 0)
             && (rocsparse::exec->count(rocsparse::execution_t::hip_free) == 0)
             && (rocsparse::exec->count(rocsparse::execution_t::hip_memset) == 0)
             && (rocsparse::exec->count(rocsparse::execution_t::hip_stream_synchronize) == 0)
             && (rocsparse::exec->count(rocsparse::execution_t::hip_device_synchronize) == 0))
            && ((rocsparse::exec->count(rocsparse::execution_t::hip_memcpy_async) > 0)
                || (rocsparse::exec->count(rocsparse::execution_t::hip_malloc_async) > 0)
                || (rocsparse::exec->count(rocsparse::execution_t::hip_free_async) > 0)
                || (rocsparse::exec->count(rocsparse::execution_t::hip_memset_async) > 0)
                || (rocsparse::exec->count(rocsparse::execution_t::hip_launch_kernel) > 0)))
               ? 1
               : 0;
}

extern "C" int32_t rocsparse_execution_is_synchronous(rocsparse_handle handle)
{
    const auto last_call = rocsparse::exec->get_last_call();
    const auto ncalls    = rocsparse::exec->get_ncalls();
    if(ncalls == 0)
        return 0;
    return ((last_call == rocsparse::execution_t::hip_memcpy)
            || (last_call == rocsparse::execution_t::hip_malloc)
            || (last_call == rocsparse::execution_t::hip_free)
            || (last_call == rocsparse::execution_t::hip_memset)
            || (last_call == rocsparse::execution_t::hip_stream_synchronize)
            || (last_call == rocsparse::execution_t::hip_device_synchronize))
               ? 1
               : 0;
}

extern "C" int32_t rocsparse_execution_is_host(rocsparse_handle handle)
{
    const auto ncalls = rocsparse::exec->get_ncalls();
    return (ncalls == 0) ? 1 : 0;
}

extern "C" void rocsparse_execution_reset(rocsparse_handle handle)
{
    rocsparse::exec->reset();
}

extern "C" int32_t rocsparse_execution_is_stream_synchronized(rocsparse_handle handle)
{

    return ((rocsparse::exec->count(rocsparse::execution_t::hip_memcpy) > 0)
            || (rocsparse::exec->count(rocsparse::execution_t::hip_malloc) > 0)
            || (rocsparse::exec->count(rocsparse::execution_t::hip_free) > 0)
            || (rocsparse::exec->count(rocsparse::execution_t::hip_memset) > 0)
            || (rocsparse::exec->count(rocsparse::execution_t::hip_stream_synchronize) > 0)
            || (rocsparse::exec->count(rocsparse::execution_t::hip_device_synchronize) > 0))
               ? 1 // Yes, it synchronized.
               : 0; // We can claim that maybe not
}

hipError_t rocsparse::execution_t::call_memcpy(void*         target,
                                               const void*   source,
                                               size_t        size,
                                               hipMemcpyKind kind)
{
    ++this->count_calls[func_t::hip_memcpy];
    return hipMemcpy(target, source, size, kind);
}

hipError_t rocsparse::execution_t::call_memcpy_async(
    void* target, const void* source, size_t size, hipMemcpyKind kind, hipStream_t stream)
{
    ++this->count_calls[func_t::hip_memcpy_async];
    return hipMemcpyAsync(target, source, size, kind, stream);
}

hipError_t rocsparse::execution_t::call_memset(void* target, int value, size_t size)
{
    ++this->count_calls[func_t::hip_memset];
    return hipMemset(target, value, size);
}

hipError_t rocsparse::execution_t::call_memset_async(void*       target,
                                                     int         value,
                                                     size_t      size,
                                                     hipStream_t stream)
{
    ++this->count_calls[func_t::hip_memset_async];
    return hipMemsetAsync(target, value, size, stream);
}

rocsparse::execution_t::func_t rocsparse::execution_t::get_last_call() const
{
    return this->m_last_call;
}

void rocsparse::execution_t::set_last_call(func_t value)
{
    this->m_last_call = value;
}

#include <iostream>
rocsparse::execution_t::~execution_t() {}

void rocsparse::execution_t::info() const
{
    const char* names[] = {"hip_malloc",
                           "hip_free",
                           "hip_malloc_async",
                           "hip_free_async",
                           "hip_memcpy",
                           "hip_memcpy_async",
                           "hip_memset",
                           "hip_memset_async",
                           "hip_stream_synchronize",
                           "hip_device_synchronize",
                           "hip_launch_kernel"};

    for(int i = 0; i < 11; ++i)
    {
        std::cout << "[" << names[i] << "] = " << count_calls[i] << std::endl;
    }
}

rocsparse::execution_t& rocsparse::execution_t::instance()
{
    static execution_t that{};
    return that;
}

void rocsparse::execution_t::flag_kernel_launch()
{
    ++this->count_calls[func_t::hip_launch_kernel];
    this->m_last_call = func_t::hip_launch_kernel;
    ++this->ncalls;
}

hipError_t rocsparse::execution_t::call_device_synchronize()
{
    ++this->count_calls[func_t::hip_device_synchronize];
    this->m_last_call = func_t::hip_device_synchronize;
    ++this->ncalls;
    return hipDeviceSynchronize();
}

hipError_t rocsparse::execution_t::call_stream_synchronize(hipStream_t stream)
{
    ++this->count_calls[func_t::hip_stream_synchronize];
    this->m_last_call = func_t::hip_stream_synchronize;
    ++this->ncalls;
    return hipStreamSynchronize(stream);
}

hipError_t rocsparse::execution_t::call_malloc_async(void** p_that, size_t size, hipStream_t stream)
{
    ++stack_count;
    ++this->count_calls[func_t::hip_malloc_async];
    this->m_last_call = func_t::hip_malloc_async;
    ++this->ncalls;
    return hipMallocAsync(p_that, size, stream);
}

hipError_t rocsparse::execution_t::call_free_async(void* that, hipStream_t stream)
{
    if(!that)
        return hipSuccess;
    --stack_count;
    ++this->count_calls[func_t::hip_free_async];
    this->m_last_call = func_t::hip_free_async;
    ++this->ncalls;
    return hipFreeAsync(that, stream);
}

hipError_t rocsparse::execution_t::call_malloc(void** p_that, size_t size)
{
    ++stack_count;
    ++this->count_calls[func_t::hip_malloc];
    this->m_last_call = func_t::hip_malloc;
    ++this->ncalls;
    return hipMalloc(p_that, size);
}

hipError_t rocsparse::execution_t::call_free(void* that)
{
    if(!that)
        return hipSuccess;
    --stack_count;
    ++this->count_calls[func_t::hip_free];
    this->m_last_call = func_t::hip_free;
    ++this->ncalls;
    return hipFree(that);
}

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
#include "rocsparse_memory_debug_t.hpp"
#include <hip/hip_runtime_api.h>
#include <iostream>

int64_t rocsparse::memory_debug_t::info_t::get_hip_ncalls() const
{
    return this->m_hip_ncalls;
}

int64_t rocsparse::memory_debug_t::info_t::get_hip_ncalls(func_t func) const
{
    return this->m_hip_count_calls[func];
}

void rocsparse::memory_debug_t::info_t::set_hip_ncalls(int64_t value)
{
    this->m_hip_ncalls = value;
}

void rocsparse::memory_debug_t::info_t::set_hip_ncalls(func_t func, int64_t value)
{
    this->m_hip_count_calls[func] = value;
}

bool rocsparse::memory_debug_t::info_t::is_hip_memory_stack_clean() const
{
    return (this->m_hip_stack_count == 0);
}

bool rocsparse::memory_debug_t::info_t::hit_hip_stream_synchronize() const
{
    return (this->m_hip_count_calls[func_t::hip_stream_synchronize] > 0);
}

bool rocsparse::memory_debug_t::info_t::hit_hip_device_synchronize() const
{
    return (this->m_hip_count_calls[func_t::hip_device_synchronize] > 0);
}

bool rocsparse::memory_debug_t::info_t::hit_hip_synchronize() const
{
    return this->hit_hip_stream_synchronize() || this->hit_hip_device_synchronize();
}

void rocsparse::memory_debug_t::info_t::reset()
{

    this->m_hip_stack_count = 0;
    this->m_hip_ncalls      = 0;
    this->m_last_hip_call   = (func_t)-1;
    for(int32_t i = 0; i < s_func_size; ++i)
    {
        this->m_hip_count_calls[i] = 0;
    }
}

void rocsparse::memory_debug_t::reset(hipStream_t stream)
{
    auto& instance = rocsparse::memory_debug_t::instance();
    instance.m_stream2info.erase(stream);
}

rocsparse::memory_debug_t::info_t& rocsparse::memory_debug_t::get_info(hipStream_t stream)
{
    return rocsparse::memory_debug_t::instance().m_stream2info[stream];
}

hipError_t rocsparse::memory_debug_t::info_t::call_memcpy2D_async(void*         target,
                                                                  size_t        tpitch,
                                                                  const void*   source,
                                                                  size_t        spitch,
                                                                  size_t        width,
                                                                  size_t        height,
                                                                  hipMemcpyKind kind,
                                                                  hipStream_t   stream)
{
    this->register_call(func_t::hip_memcpy2d_async);
    auto e = hipMemcpy2DAsync(target, tpitch, source, spitch, width, height, kind, stream);
    if(e != hipSuccess)
    {
        return e;
    }
    this->add_data_transfer(width * height);
    return e;
}

hipError_t rocsparse::memory_debug_t::info_t::call_memcpy(void*         target,
                                                          const void*   source,
                                                          size_t        size,
                                                          hipMemcpyKind kind)
{
    this->register_call(func_t::hip_memcpy);
    auto e = hipMemcpy(target, source, size, kind);
    if(e != hipSuccess)
    {
        return e;
    }
    this->add_data_transfer(size);
    return e;
}

hipError_t rocsparse::memory_debug_t::info_t::call_memcpy_async(
    void* target, const void* source, size_t size, hipMemcpyKind kind, hipStream_t stream)
{
    this->register_call(func_t::hip_memcpy_async);
    auto e = hipMemcpyAsync(target, source, size, kind, stream);
    if(e != hipSuccess)
    {
        return e;
    }
    this->add_data_transfer(size);
    return e;
}

hipError_t rocsparse::memory_debug_t::info_t::call_memset(void* target, int value, size_t size)
{
    this->register_call(func_t::hip_memset);
    return hipMemset(target, value, size);
}

void rocsparse::memory_debug_t::info_t::add_data_transfer(size_t size_in_bytes)
{
    const double delta    = double(size_in_bytes) / double(1024 * 1024 * 1024);
    double       expected = this->m_gib.load(std::memory_order_relaxed);
    while(!this->m_gib.compare_exchange_weak(
        expected, expected + delta, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }
}

void rocsparse::memory_debug_t::info_t::register_call(func_t f)
{
    this->set_last_hip_call(f);
    this->m_hip_count_calls[f].fetch_add(1, std::memory_order_relaxed);
    this->m_hip_ncalls.fetch_add(1, std::memory_order_relaxed);
}

hipError_t rocsparse::memory_debug_t::info_t::call_memset_async(void*       target,
                                                                int         value,
                                                                size_t      size,
                                                                hipStream_t stream)
{
    this->register_call(func_t::hip_memset_async);
    return hipMemsetAsync(target, value, size, stream);
}

rocsparse::memory_debug_t::func_t rocsparse::memory_debug_t::info_t::get_last_hip_call() const
{
    return this->m_last_hip_call;
}

void rocsparse::memory_debug_t::info_t::set_last_hip_call(func_t value)
{
    this->m_last_hip_call = value;
}

rocsparse::memory_debug_t::~memory_debug_t() {}

void rocsparse::memory_debug_t::info_t::info() const
{
    const char* names[] = {"hip_malloc",
                           "hip_free",
                           "hip_malloc_async",
                           "hip_free_async",
                           "hip_memcpy",
                           "hip_memcpy_async",
                           "hip_memcpy2D_async",
                           "hip_memset",
                           "hip_memset_async",
                           "hip_stream_synchronize",
                           "hip_device_synchronize",
                           "hip_launch_kernel"};

    for(int32_t i = 0; i < s_func_size; ++i)
    {
        std::cout << "[" << names[i] << "] = " << this->m_hip_count_calls[i] << std::endl;
    }
}

rocsparse::memory_debug_t& rocsparse::memory_debug_t::instance()
{
    static memory_debug_t that{};
    return that;
}

void rocsparse::memory_debug_t::info_t::flag_hip_launch_kernel()
{
    this->register_call(func_t::hip_launch_kernel);
}

hipError_t rocsparse::memory_debug_t::info_t::call_device_synchronize()
{
    this->register_call(func_t::hip_device_synchronize);
    return hipDeviceSynchronize();
}

hipError_t rocsparse::memory_debug_t::info_t::call_stream_synchronize(hipStream_t stream)
{
    this->register_call(func_t::hip_stream_synchronize);
    return hipStreamSynchronize(stream);
}

hipError_t rocsparse::memory_debug_t::info_t::call_malloc_async(void**      p_that,
                                                                size_t      size,
                                                                hipStream_t stream)
{
    ++this->m_hip_stack_count;
    this->register_call(func_t::hip_malloc_async);

// if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
    auto e = hipMallocAsync(p_that, size, stream);
#else
    auto e = hipMalloc(p_that, size);
#endif
    if(e != hipSuccess)
    {
        --this->m_hip_stack_count;
    }
    return e;
}

hipError_t rocsparse::memory_debug_t::info_t::call_free_async(void* that, hipStream_t stream)
{
    if(!that)
        return hipSuccess;
    --this->m_hip_stack_count;
    this->register_call(func_t::hip_free_async);
    // if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
    return hipFreeAsync(that, stream);
#else
    return hipFree(that);
#endif
}

hipError_t rocsparse::memory_debug_t::info_t::call_malloc(void** p_that, size_t size)
{
    this->register_call(func_t::hip_malloc);
    ++this->m_hip_stack_count;
    auto e = hipMalloc(p_that, size);
    if(e != hipSuccess)
    {
        --this->m_hip_stack_count;
    }
    return e;
}

hipError_t rocsparse::memory_debug_t::info_t::call_free(void* that)
{
    if(!that)
        return hipSuccess;
    this->register_call(func_t::hip_free);
    --this->m_hip_stack_count;
    return hipFree(that);
}

double rocsparse::memory_debug_t::info_t::get_data_transfer_in_gib() const
{
    return this->m_gib;
}

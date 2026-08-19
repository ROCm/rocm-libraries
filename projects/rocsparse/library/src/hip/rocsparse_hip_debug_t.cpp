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
#include "rocsparse_hip_debug_t.hpp"
#include "rocsparse_handle.hpp"
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <mutex>
#define ROCSPARSE_MUTEX 1
using api_history_t = rocsparse::hip::debug_t::api_history_t;
using api_t         = rocsparse::hip::debug_t::api_t;

int64_t api_history_t::get_hip_ncalls() const
{
    return this->m_hip_ncalls;
}

int64_t api_history_t::get_hip_ncalls(api_t::type_t func) const
{
    return this->m_hip_count_calls[func];
}

void api_history_t::set_hip_ncalls(int64_t value)
{
    this->m_hip_ncalls = value;
}

void api_history_t::set_hip_ncalls(api_t::type_t func, int64_t value)
{
    this->m_hip_count_calls[func] = value;
}

bool api_history_t::is_hip_memory_stack_clean() const
{
    return (this->m_hip_stack_count == 0);
}

bool api_history_t::hit_hip_stream_synchronize() const
{
    return (this->m_hip_count_calls[api_t::hipStreamSynchronize] > 0);
}

bool api_history_t::hit_hip_device_synchronize() const
{
    return (this->m_hip_count_calls[api_t::hipDeviceSynchronize] > 0);
}

bool api_history_t::hit_hip_synchronize() const
{
    return this->hit_hip_stream_synchronize() || this->hit_hip_device_synchronize();
}

void api_history_t::reset()
{
    this->m_hip_stack_count = 0;
    this->m_hip_ncalls      = 0;
    this->m_last_hip_call   = (api_t::type_t)-1;
    for(int32_t i = 0; i < api_t::size; ++i)
    {
        this->m_hip_count_calls[i] = 0;
    }
    this->m_gib = 0;
    this->m_api_history.clear();
}

hipError_t
    rocsparse::hip::debug_t::hipMemcpy2DAsync(void*                                        target,
                                              size_t                                       tpitch,
                                              const void*                                  source,
                                              size_t                                       spitch,
                                              size_t                                       width,
                                              size_t                                       height,
                                              hipMemcpyKind                                kind,
                                              hipStream_t                                  stream,
                                              const rocsparse::hip::debug_t::code_trace_t& trace)
{
    auto* h = rocsparse::hip::debug_t::find_api_history(stream);
    h->register_call(api_t::hipMemcpy2DAsync, stream, trace);
    h->add_data_transfer(width * height);
    return ::hipMemcpy2DAsync(target, tpitch, source, spitch, width, height, kind, stream);
}

hipError_t rocsparse::hip::debug_t::hipMemcpy(void*                                        target,
                                              const void*                                  source,
                                              size_t                                       size,
                                              hipMemcpyKind                                kind,
                                              const rocsparse::hip::debug_t::code_trace_t& trace)
{
    hipStream_t default_stream{};
    auto*       h = rocsparse::hip::debug_t::find_api_history(default_stream);
    h->register_call(api_t::hipMemcpy, default_stream, trace);
    auto e = ::hipMemcpy(target, source, size, kind);
    if(e != hipSuccess)
    {
        return e;
    }
    h->add_data_transfer(size);
    return hipSuccess;
}

hipError_t
    rocsparse::hip::debug_t::hipMemcpyAsync(void*                                        target,
                                            const void*                                  source,
                                            size_t                                       size,
                                            hipMemcpyKind                                kind,
                                            hipStream_t                                  stream,
                                            const rocsparse::hip::debug_t::code_trace_t& trace)
{
    auto* h = rocsparse::hip::debug_t::find_api_history(stream);
    h->register_call(api_t::hipMemcpyAsync, stream, trace);
    h->add_data_transfer(size);
    return ::hipMemcpyAsync(target, source, size, kind, stream);
}

hipError_t rocsparse::hip::debug_t::hipMemset(void*                                        target,
                                              int                                          value,
                                              size_t                                       size,
                                              const rocsparse::hip::debug_t::code_trace_t& trace)
{
    hipStream_t default_stream{};
    rocsparse::hip::debug_t::register_call(api_t::hipMemset, default_stream, trace);
    return ::hipMemset(target, value, size);
}

void api_history_t::add_data_transfer(size_t size_in_bytes)
{
    const double delta = double(size_in_bytes) / double(1024 * 1024 * 1024);
    this->m_gib += delta;
}

void api_history_t::register_call(api_t::type_t                                f,
                                  hipStream_t                                  stream,
                                  const rocsparse::hip::debug_t::code_trace_t& trace)
{
    this->m_api_history.push_back({f, stream, trace});
    this->m_hip_count_calls[f] += 1;
    this->m_hip_ncalls += 1;
}

void rocsparse::hip::debug_t::register_call(api_t::type_t                                f,
                                            hipStream_t                                  stream,
                                            const rocsparse::hip::debug_t::code_trace_t& trace)
{
    auto* h = rocsparse::hip::debug_t::find_api_history(stream);
    h->register_call(f, stream, trace);
}

hipError_t
    rocsparse::hip::debug_t::hipMemsetAsync(void*                                        target,
                                            int                                          value,
                                            size_t                                       size,
                                            hipStream_t                                  stream,
                                            const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipMemsetAsync, stream, trace);
    return ::hipMemsetAsync(target, value, size, stream);
}

api_t::type_t api_history_t::get_last_hip_call() const
{

    auto* h = rocsparse::hip::debug_t::find_api_history((rocsparse_handle) nullptr);
    if(h->m_api_history.size() > 0)
    {
        return h->m_api_history.back().m_func;
    }
    return rocsparse::hip::debug_t::api_t::unknown;
}

void rocsparse::hip::debug_t::tag_hipLaunchKernelGGL(
    hipStream_t stream, const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipLaunchKernelGGL, stream, trace);
}

hipError_t rocsparse::hip::debug_t::hipDeviceSynchronize(
    const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipDeviceSynchronize, nullptr, trace);
    return ::hipDeviceSynchronize();
}

hipError_t rocsparse::hip::debug_t::hipStreamSynchronize(
    hipStream_t stream, const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipStreamSynchronize, stream, trace);
    return ::hipStreamSynchronize(stream);
}

hipError_t
    rocsparse::hip::debug_t::hipMallocAsync(void**                                       p_that,
                                            size_t                                       size,
                                            hipStream_t                                  stream,
                                            const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipMallocAsync, stream, trace);

    // if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
    auto error = ::hipMallocAsync(p_that, size, stream);
#else
    auto error = ::hipMalloc(p_that, size);
#endif
    return error;
}

hipError_t rocsparse::hip::debug_t::hipFreeAsync(void*       that,
                                                 hipStream_t stream,
                                                 const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipFreeAsync, stream, trace);
    // if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
    return ::hipFreeAsync(that, stream);
#else
    return ::hipFree(that);
#endif
}

hipError_t rocsparse::hip::debug_t::hipMalloc(void**                                       p_that,
                                              size_t                                       size,
                                              const rocsparse::hip::debug_t::code_trace_t& trace)
{
    rocsparse::hip::debug_t::register_call(api_t::hipMalloc, nullptr, trace);
    return ::hipMalloc(p_that, size);
}

hipError_t rocsparse::hip::debug_t::hipFree(void*                                        that,
                                            const rocsparse::hip::debug_t::code_trace_t& trace)
{
    static hipStream_t default_stream{};
    rocsparse::hip::debug_t::register_call(api_t::hipFree, default_stream, trace);
    return ::hipFree(that);
}

double api_history_t::get_data_transfer_in_gib() const
{
    return this->m_gib;
}

rocsparse::hip::debug_t& rocsparse::hip::debug_t::instance()
{
    static hip::debug_t that{};
    return that;
}

static rocsparse::hip::debug_t::key_t make_key(rocsparse_handle handle)
{
    int32_t gpu_id;
    THROW_IF_HIP_ERROR(hipGetDevice(&gpu_id));
    return {std::this_thread::get_id(), gpu_id};
}

static rocsparse::hip::debug_t::key_t make_key(hipStream_t stream)
{
    int32_t gpu_id;
    THROW_IF_HIP_ERROR(hipGetDevice(&gpu_id));
    return {std::this_thread::get_id(), gpu_id};
}

void rocsparse::hip::debug_t::reset(rocsparse_handle handle)
{
    auto& instance = rocsparse::hip::debug_t::instance();
    auto  key      = make_key(handle);
#if ROCSPARSE_MUTEX
    std::unique_lock lock(instance.shared_mutex);
#endif
    instance.m_api_history[key].reset();
}

api_history_t* rocsparse::hip::debug_t::find_api_history(rocsparse_handle handle)
{
    auto& instance = rocsparse::hip::debug_t::instance();
#if ROCSPARSE_MUTEX
    std::shared_lock lock(instance.shared_mutex);
#endif
    auto key = make_key(handle);
    auto it  = instance.m_api_history.find(key);
    return (it == instance.m_api_history.end()) ? nullptr : &it->second;
}

bool rocsparse::hip::debug_t::enabled()
{
    const auto& instance = rocsparse::hip::debug_t::instance();
#if ROCSPARSE_MUTEX
    std::shared_lock lock(instance.shared_mutex);
#endif
    return instance.m_enable;
}
void rocsparse::hip::debug_t::disable()
{
    auto& instance = rocsparse::hip::debug_t::instance();
#if ROCSPARSE_MUTEX
    std::unique_lock lock(instance.shared_mutex);
#endif
    instance.m_enable = false;
}
void rocsparse::hip::debug_t::enable()
{
    auto& instance = rocsparse::hip::debug_t::instance();
#if ROCSPARSE_MUTEX
    std::unique_lock lock(instance.shared_mutex);
#endif
    instance.m_enable = true;
}

api_history_t* rocsparse::hip::debug_t::find_api_history(hipStream_t stream)
{
    auto& instance = rocsparse::hip::debug_t::instance();
#if ROCSPARSE_MUTEX
    std::unique_lock lock(instance.shared_mutex);
#endif
    auto key = make_key(stream);
    auto it  = instance.m_api_history.find(key);
    if(it == instance.m_api_history.end())
    {
        return &instance.m_api_history[key];
    }
    else
    {
        return &it->second;
    }
}

rocsparse::hip::debug_t::~debug_t() {}

std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t::api_t::type_t& value)
{
    switch(value)
    {
    case rocsparse::hip::debug_t::api_t::hipMallocAsync:
    {
        os << "hipMallocAsync";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipFreeAsync:
    {
        os << "hipFreeAsync";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMemcpyAsync:
    {
        os << "hipMemcpyAsync";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMemcpy2DAsync:
    {
        os << "hipMemcpy2dAsync";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMemsetAsync:
    {
        os << "hipMemsetAsync";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipLaunchKernelGGL:
    {
        os << "hipLaunchKernelGGL";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMalloc:
    {
        os << "hipMalloc";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipFree:
    {
        os << "hipFree";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMemcpy:
    {
        os << "hipMemcpy";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipMemset:
    {
        os << "hipMemset";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipStreamSynchronize:
    {
        os << "hipStreamSynchronize";
        break;
    }

    case rocsparse::hip::debug_t::api_t::hipDeviceSynchronize:
    {
        os << "hipDeviceSynchronize";
        break;
    }

    case rocsparse::hip::debug_t::api_t::unknown:
    {
        os << "UNKNOWN";
        break;
    }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t::code_trace_t& value)
{
    os << "func :" << value.m_function << std::endl;
    os << "file :" << value.m_filename << std::endl;
    os << "line :" << value.m_line;
    return os;
}

std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t::api_history_item_t& value)
{
    os << "api : " << value.m_func << std::endl;
    os << "stream : " << value.m_stream << std::endl;
    os << "trace  : " << value.m_code_trace;
    return os;
}

std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t::api_history_t& value)
{
    os << "history :" << std::endl;
    for(const auto& item : value.m_api_history)
    {
        std::cout << " " << item << std::endl;
    }
    os << "history done";
    return os;
}

std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t::key_t& key)
{
    os << "> tid              : " << std::get<std::thread::id>(key) << std::endl;
    os << "> gpuid            : " << std::get<int32_t>(key);
    return os;
}
std::ostream& operator<<(std::ostream& os, const rocsparse::hip::debug_t& value)
{
    os << "debug :" << std::endl;
    for(const auto& it : value.m_api_history)
    {
        const auto& key = it.first;
        const auto& h   = it.second;
        os << "tid and gpu_id: " << std::endl
           << key << std::endl
           << "api_history: " << std::endl
           << h << std::endl;
    }
    os << "debug done";
    return os;
}

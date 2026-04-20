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
#include "rocsparse-debugging.h"
#include "rocsparse_handle.hpp"
#include "rocsparse_memory_debug_t.hpp"
#include <iostream>

namespace rocsparse
{
    static bool is_blocking(memory_debug_t::func_t f)
    {
        switch(f)
        {
        case memory_debug_t::func_t::hip_malloc_async:
        case memory_debug_t::func_t::hip_free_async:
        case memory_debug_t::func_t::hip_memcpy_async:
        case memory_debug_t::func_t::hip_memcpy2d_async:
        case memory_debug_t::func_t::hip_memset_async:
        case memory_debug_t::func_t::hip_launch_kernel:
        {
            return false;
        }
        case memory_debug_t::func_t::hip_malloc:
        case memory_debug_t::func_t::hip_free:
        case memory_debug_t::func_t::hip_memcpy:
        case memory_debug_t::func_t::hip_memset:
        case memory_debug_t::func_t::hip_stream_synchronize:
        case memory_debug_t::func_t::hip_device_synchronize:
        {
            return true;
        }
        }
    }

    static inline bool is_blocking(const memory_debug_t::info_t& info)
    {
        return ((info.get_hip_ncalls(memory_debug_t::hip_memcpy) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_malloc) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_free) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_memset) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_stream_synchronize) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_device_synchronize) > 0));
    }

    static inline bool is_non_blocking(const memory_debug_t::info_t& info)
    {
        return ((info.get_hip_ncalls(memory_debug_t::hip_memcpy_async) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_memcpy2d_async) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_malloc_async) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_free_async) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_launch_kernel) > 0)
                || (info.get_hip_ncalls(memory_debug_t::hip_memset_async) > 0));
    }

}

extern "C" rocsparse_status rocsparse_memory_debug_reset(rocsparse_handle handle)
{
    hipStream_t default_stream{};
    rocsparse::memory_debug_t::reset((handle) ? handle->stream : default_stream);
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_memory_debug_info_get(rocsparse_handle            handle,
                                                            rocsparse_memory_debug_info debug_info,
                                                            void*                       data,
                                                            size_t data_size_in_bytes)
{
    hipStream_t default_stream{};
    auto& info = rocsparse::memory_debug_t::get_info((handle) ? handle->stream : default_stream);
    switch(debug_info)
    {
    case rocsparse_memory_debug_info_synchronicity:
    {
        auto p_data = reinterpret_cast<rocsparse_memory_debug_synchronicity*>(data);
        if(rocsparse::is_blocking(info))
        {
            const auto last_hip_call = info.get_last_hip_call();
            p_data[0] = (is_blocking(last_hip_call)) ? rocsparse_memory_debug_synchronicity_sync
                                                     : rocsparse_memory_debug_synchronicity_psync;
        }
        else if(rocsparse::is_non_blocking(info))
        {
            p_data[0] = rocsparse_memory_debug_synchronicity_async;
        }
        else
        {
            p_data[0] = rocsparse_memory_debug_synchronicity_host;
        }

        return rocsparse_status_success;
    }

    case rocsparse_memory_debug_info_transfer_nbytes:
    {
        *reinterpret_cast<double*>(data) = info.get_data_transfer_in_gib();
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

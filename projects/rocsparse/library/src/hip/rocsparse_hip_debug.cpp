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
#include "rocsparse_hip_debug_t.hpp"
#include <iostream>

namespace rocsparse
{
    namespace hip
    {
        static bool is_hipYYYAsync(hip::debug_t::api_t::type_t f)
        {
            switch(f)
            {
            case hip::debug_t::api_t::hipMallocAsync:
            case hip::debug_t::api_t::hipFreeAsync:
            case hip::debug_t::api_t::hipMemcpyAsync:
            case hip::debug_t::api_t::hipMemcpy2DAsync:
            case hip::debug_t::api_t::hipMemsetAsync:
            case hip::debug_t::api_t::hipLaunchKernelGGL:
            {
                return true;
            }
            case hip::debug_t::api_t::hipMalloc:
            case hip::debug_t::api_t::hipFree:
            case hip::debug_t::api_t::hipMemcpy:
            case hip::debug_t::api_t::hipMemset:
            case hip::debug_t::api_t::hipStreamSynchronize:
            case hip::debug_t::api_t::hipDeviceSynchronize:
            {
                return false;
            }
            case hip::debug_t::api_t::unknown:
            {
                THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_internal_error,
                    "hip::debug_t::api_t::unknown means it the variable has not been initialized.");
            }
            }
        }

        static inline bool is_hipYYYAsync(const hip::debug_t::api_history_t& that)
        {
            const auto& h = that.m_api_history;
            for(const auto& item : h)
            {
                //	  auto stream = item.m_stream;
                auto f = item.m_func;
                if(!rocsparse::hip::is_hipYYYAsync(f))
                {
                    return false;
                }
            }
            return (h.size() > 0) ? true : false;
        }

        static inline bool any(const hip::debug_t::api_history_t* that)
        {
            return (that->m_api_history.size() > 0);
        }
    }
}

extern "C" rocsparse_status
    rocsparse_hip_debug_api_info_get(rocsparse_handle             handle,
                                     rocsparse_hip_debug_api      debug_tag,
                                     rocsparse_hip_debug_api_info debug_tag_info,
                                     void*                        data,
                                     size_t                       data_size_in_bytes,
                                     rocsparse_error*             p_error)
{
    auto* history = rocsparse::hip::debug_t::find_api_history(handle);
    switch(debug_tag_info)
    {
    case rocsparse_hip_debug_api_info_count:
    {
        *reinterpret_cast<int64_t*>(data)
            = history->get_hip_ncalls((rocsparse::hip::debug_t::api_t::type_t)debug_tag);
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

extern "C" rocsparse_status rocsparse_hip_debug_enable()
{
    rocsparse::hip::debug_t::enable();
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_hip_debug_disable()
{
    rocsparse::hip::debug_t::disable();
    return rocsparse_status_success;
}

extern "C" int32_t rocsparse_hip_debug_state()
{
    return rocsparse::hip::debug_t::enabled() ? 1 : 0;
}

extern "C" rocsparse_status rocsparse_hip_debug_start(rocsparse_handle handle,
                                                      rocsparse_error* p_error)
{
    rocsparse::hip::debug_t::reset(handle);
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_hip_debug_print(rocsparse_handle handle,
                                                      rocsparse_error* p_error)
{
    const auto& instance = rocsparse::hip::debug_t::instance();
    std::cout << instance << std::endl;
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_hip_debug_info_get(rocsparse_handle         handle,
                                                         rocsparse_hip_debug_info debug_info,
                                                         void*                    data,
                                                         size_t           data_size_in_bytes,
                                                         rocsparse_error* p_error)
{
    ROCSPARSE_CHECKARG_POINTER(2, data);

    auto* history = rocsparse::hip::debug_t::find_api_history(handle);
    switch(debug_info)
    {
    case rocsparse_hip_debug_info_api:
    {
        if(data_size_in_bytes != sizeof(rocsparse_hip_debug_api_history))
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
        }

        auto p_data = reinterpret_cast<rocsparse_hip_debug_api_history*>(data);
        if(rocsparse::hip::any(history))
        {
            if(rocsparse::hip::is_hipYYYAsync(history[0]))
            {
                p_data[0] = rocsparse_hip_debug_api_history_async;
            }
            else if(rocsparse::hip::is_hipYYYAsync(history[0].get_last_hip_call()))
            {
                p_data[0] = rocsparse_hip_debug_api_history_psync;
            }
            else
            {
                p_data[0] = rocsparse_hip_debug_api_history_sync;
            }
        }
        else
        {
            p_data[0] = rocsparse_hip_debug_api_history_none;
        }

        return rocsparse_status_success;
    }

    case rocsparse_hip_debug_info_transfer_in_gib:
    {
        if(data_size_in_bytes != sizeof(double))
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
        }
        *reinterpret_cast<double*>(data) = history[0].get_data_transfer_in_gib();
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

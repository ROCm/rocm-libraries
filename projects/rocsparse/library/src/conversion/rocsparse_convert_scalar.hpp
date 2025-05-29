/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"
#include "to_string.hpp"

namespace rocsparse
{
    typedef void (*convert_scalar_t)(const void*, void*);

    convert_scalar_t find_convert_scalar(rocsparse_datatype source_datatype,
                                         rocsparse_datatype target_datatype);

    template <typename... P>
    inline void convert_host_scalars_impl(convert_scalar_t convert_scalar)
    {
    }

    template <typename... P>
    inline void convert_host_scalars_impl(convert_scalar_t convert_scalar,
                                          const void*      source,
                                          void*            target,
                                          P... p)
    {
        convert_scalar(source, target);
        convert_host_scalars_impl(convert_scalar, p...);
    }

    template <typename... P>
    inline rocsparse_status convert_host_scalars(rocsparse_datatype source_datatype,
                                                 rocsparse_datatype target_datatype,
                                                 const void*        source,
                                                 void*              target,
                                                 P... p)
    {
        convert_scalar_t convert_scalar = find_convert_scalar(source_datatype, target_datatype);

        if(convert_scalar != nullptr)
        {
            convert_host_scalars_impl(convert_scalar, source, target, p...);
            return rocsparse_status_success;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "source_datatype: " << rocsparse::to_string(source_datatype)
                 << "target_datatype: " << rocsparse::to_string(target_datatype);
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP
    }

    template <typename F>
    __device__ inline void convert_device_scalars_kernel_device(F f, uint32_t tid)
    {
    }

    template <typename F, typename... P>
    __device__ inline void convert_device_scalars_kernel_device(
        F f, uint32_t tid, const void* source, void* target, P... p)
    {
        if(tid == 0)
        {
            f(source, target);
        }
        convert_device_scalars_kernel_device(f, tid, p...);
    }

    template <uint32_t BLOCKSIZE, typename F, typename... P>
    __launch_bounds__(BLOCKSIZE) __global__
        void convert_device_scalars_kernel(F f, const void* source, void* target, P... p)
    {
        const size_t tid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        convert_device_scalars_kernel_device(f, tid, source, target, p...);
    }

    template <typename... P>
    inline rocsparse_status convert_device_scalars(hipStream_t        stream,
                                                   rocsparse_datatype source_datatype,
                                                   rocsparse_datatype target_datatype,
                                                   const void*        source,
                                                   void*              target,
                                                   P... p)
    {
        convert_scalar_t convert_scalar = find_convert_scalar(source_datatype, target_datatype);

        if(convert_scalar != nullptr)
        {
            static uint32_t blocksize = 32;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(convert_device_scalars_kernel<32>,
                                               dim3(1),
                                               dim3(blocksize),
                                               0,
                                               stream,
                                               convert_scalar,
                                               source,
                                               target,
                                               p...);
            return rocsparse_status_success;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "source_datatype: " << rocsparse::to_string(source_datatype)
                 << "target_datatype: " << rocsparse::to_string(target_datatype);
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP
    }

}

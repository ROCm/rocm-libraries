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
#include "rocsparse_enum_utils.hpp"

namespace rocsparse
{
    typedef void (*host_convert_scalar_t)(const void*, void*);
    typedef void (*device_convert_scalar_t)(const void*, void*);

    host_convert_scalar_t find_host_convert_scalar(rocsparse_datatype source_datatype,
                                                   rocsparse_datatype target_datatype);

    device_convert_scalar_t find_device_convert_scalar(rocsparse_datatype source_datatype,
                                                       rocsparse_datatype target_datatype);

    template <typename... P>
    inline void convert_host_scalars_impl(host_convert_scalar_t convert_scalar)
    {
    }

    template <typename... P>
    inline void convert_host_scalars_impl(host_convert_scalar_t convert_scalar,
                                          const void*           source,
                                          void*                 target,
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
        host_convert_scalar_t convert_scalar
            = find_host_convert_scalar(source_datatype, target_datatype);

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
                 << "source_datatype: " << rocsparse::enum_utils::to_string(source_datatype)
                 << "target_datatype: " << rocsparse::enum_utils::to_string(target_datatype);
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    inline rocsparse_status convert_device_scalars(hipStream_t        stream,
                                                   rocsparse_datatype source_datatype,
                                                   rocsparse_datatype target_datatype,
                                                   const void*        source,
                                                   void*              target,
                                                   P... p)
    {
        device_convert_scalar_t convert_scalar
            = find_device_convert_scalar(source_datatype, target_datatype);

        static constexpr int32_t n = (2 + sizeof...(p)) / 2;
        const void*              sources[n];
        void*                    targets[n];

        {
            struct st_t
            {
                const void* s;
                void*       t;
            };

            const void* arg[] = {source, target, p...};
            for(int i = 0; i < n; ++i)
            {
                sources[i] = ((st_t*)arg)[i].s;
                targets[i] = ((st_t*)arg)[i].t;
            }
        }

        if(convert_scalar != nullptr)
        {
            static uint32_t blocksize = 64;
            for(int i = 0; i < n; ++i)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    convert_scalar, dim3(1), dim3(blocksize), 0, stream, sources[i], targets[i]);
            }
            return rocsparse_status_success;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "source_datatype: " << rocsparse::enum_utils::to_string(source_datatype)
                 << "target_datatype: " << rocsparse::enum_utils::to_string(target_datatype);
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP
    }

}

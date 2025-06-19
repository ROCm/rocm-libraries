/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <typename T>
    union const_host_device_scalar
    {
        T                        value;
        const T*                 pointer;
        __forceinline__ __host__ const_host_device_scalar(const T* scalar)
            : pointer(scalar){};
        __forceinline__ __host__ const_host_device_scalar(const T& scalar)
            : value(scalar){};
        static __forceinline__ __device__ T zero()
        {
            return static_cast<T>(0);
        }
    };

    template <typename T>
    __forceinline__ __host__ rocsparse::const_host_device_scalar<T>
                             to_const_host_device_scalar(rocsparse_pointer_mode mode, const T* s_)
    {
        return (mode == rocsparse_pointer_mode_host) ? rocsparse::const_host_device_scalar<T>(*s_)
                                                     : rocsparse::const_host_device_scalar<T>(s_);
    }

    template <typename T>
    __forceinline__ __host__ rocsparse::const_host_device_scalar<T>
        to_permissive_const_host_device_scalar(rocsparse_pointer_mode mode, const T* s_)
    {
        return (mode == rocsparse_pointer_mode_host)
                   ? rocsparse::const_host_device_scalar<T>((s_) ? *s_ : T(0))
                   : rocsparse::const_host_device_scalar<T>(s_);
    }

}

#define ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(H_, S_) \
    rocsparse::to_const_host_device_scalar((H_)->pointer_mode, S_)

#define ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(H_, S_) \
    rocsparse::to_permissive_const_host_device_scalar((H_)->pointer_mode, S_)

#define ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(TYPE_, NAME_) \
    rocsparse::const_host_device_scalar<TYPE_> NAME_##_union

#define ROCSPARSE_DEVICE_HOST_SCALAR_GET(NAME_) \
    const auto NAME_ = (is_host_mode) ? NAME_##_union.value : *NAME_##_union.pointer

#define ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(COND_, NAME_)                                        \
    const auto NAME_ = (COND_) ? ((is_host_mode) ? NAME_##_union.value : *NAME_##_union.pointer) \
                               : NAME_##_union.zero()

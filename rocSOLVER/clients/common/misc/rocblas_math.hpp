/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <type_traits>

/* ============================================================================================
 */
// Helper function to truncate float to bfloat16

inline __host__ rocblas_bfloat16 float_to_bfloat16_truncate(float val)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {val};
    rocblas_bfloat16 ret;
    ret.data = uint16_t(u.int32 >> 16);
    if((u.int32 & 0x7fff0000) == 0x7f800000 && u.int32 & 0xffff)
        ret.data |= 1; // Preserve signaling NaN
    return ret;
}

/* ============================================================================================
 */
/*! \brief  returns true if value is NaN */

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
inline bool rocblas_isnan(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{} && !rocblas_is_complex<T>, int> = 0>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

inline bool rocblas_isnan(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/* ============================================================================================
 */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

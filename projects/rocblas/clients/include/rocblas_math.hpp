/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocblas_xfloat32.h"
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>

/* ============================================================================================ */
// Helper function to truncate float to bfloat16

inline __host__ rocblas_bfloat16 float_to_bfloat16_truncate(float val)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {val};
    rocblas_bfloat16 ret;
    ret.data = uint16_t(u.int32 >> 16);
    if((u.int32 & 0x7fff0000) == 0x7f800000 && u.int32 & 0xffff)
        ret.data |= 1; // Preserve signaling NaN
    return ret;
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
__device__ __host__ inline T negate(T x)
{
    return -x;
}

template <>
__device__ __host__ inline rocblas_half negate(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
__device__ __host__ inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

/* ============================================================================================ */
// Helper function to reduce intermediate precision and the output type are the same as the input type.
template <typename TxDLi, typename TxDLo, typename Ti>
inline void type_to_xdl_math_op_type(Ti* in, size_t s)
{
    //To filter out the case that input type is not supported by xDL Math Op.
    //Currently, xDL Math Op supports in:float -> intermediat:xf32 -> out:float
    constexpr bool needCast = !std::is_same<TxDLi, Ti>() && std::is_same<TxDLo, Ti>();
    if(!needCast)
        return;

    //Cast input type to xDl math op type, using type alians to avoid the casting error.
    using castType = std::conditional_t<needCast, TxDLi, Ti>;
    for(size_t i = 0; i < s; i++)
        in[i] = static_cast<Ti>(static_cast<castType>(in[i]));
}

// Conjugate a value. For most types, simply return argument; for
// rocblas_float_complex and rocblas_double_complex, return std::conj(z)
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline T conjugate(const T& z)
{
    return z;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline T conjugate(const T& z)
{
    return std::conj(z);
}

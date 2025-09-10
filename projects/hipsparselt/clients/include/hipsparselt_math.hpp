/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "hipsparselt_fp8.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>
#include <immintrin.h>
#include <type_traits>

/* ============================================================================================ */
// Helper function to truncate float to bfloat16

inline __host__ hip_bfloat16 float_to_bfloat16_truncate(float val)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hip_bfloat16(val, hip_bfloat16::truncate_t::truncate);
#else
    return __float2bfloat16_rd(val);
#endif
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline __half negate(__half arg)
{
    auto raw = __half_raw(arg);
    raw.x ^= 0x8000;
    return raw;
}

template <>
inline hip_bfloat16 negate(hip_bfloat16 x)
{

#if defined(__HIP_PLATFORM_AMD__)
    x.data ^= 0x8000;
    return x;
#else
    auto raw = __nv_bfloat16_raw(x);
    raw.x ^= 0x8000;
    return raw;
#endif
}

#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
template <>
inline hipsparselt_fp8_e4m3 negate(hipsparselt_fp8_e4m3 x)
{
    x.__x ^= 0x80;
    return x;
}

template <>
inline hipsparselt_fp8_e5m2 negate(hipsparselt_fp8_e5m2 x)
{
    x.__x ^= 0x80;
    return x;
}
#endif

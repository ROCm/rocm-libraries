/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <hip/hip_bfloat16.h>
#include "vector_types.hpp"

namespace miopen {

//=============================================================================
// Float overloads
//=============================================================================

__forceinline__ __device__ float exp(float x) { return expf(x); }
__forceinline__ __device__ float log(float x) { return logf(x); }
__forceinline__ __device__ float sqrt(float x) { return sqrtf(x); }
__forceinline__ __device__ float rsqrt(float x) { return rsqrtf(x); }
__forceinline__ __device__ float sin(float x) { return sinf(x); }
__forceinline__ __device__ float cos(float x) { return cosf(x); }
__forceinline__ __device__ float tan(float x) { return tanf(x); }
__forceinline__ __device__ float tanh(float x) { return tanhf(x); }
__forceinline__ __device__ float pow(float x, float y) { return powf(x, y); }
__forceinline__ __device__ float fabs(float x) { return fabsf(x); }
__forceinline__ __device__ float fmax(float x, float y) { return fmaxf(x, y); }
__forceinline__ __device__ float fmin(float x, float y) { return fminf(x, y); }
__forceinline__ __device__ float fma(float a, float b, float c) { return ::fma(a, b, c); }
__forceinline__ __device__ bool isnan(float a) { return ::isnan(a); }
__forceinline__ __device__ bool isinf(float a) { return ::isinf(a); }

//=============================================================================
// Half precision overloads
//=============================================================================

__forceinline__ __device__ _Float16 exp(_Float16 x) { return __ocml_exp_f16(x); }
__forceinline__ __device__ _Float16 log(_Float16 x) { return __ocml_log_f16(x); }
__forceinline__ __device__ _Float16 sqrt(_Float16 x) { return __ocml_sqrt_f16(x); }
__forceinline__ __device__ _Float16 rsqrt(_Float16 x) { return __ocml_rsqrt_f16(x); }
__forceinline__ __device__ _Float16 sin(_Float16 x) { return hsin(__half(x)); }
__forceinline__ __device__ _Float16 cos(_Float16 x) { return hcos(__half(x)); }
__forceinline__ __device__ _Float16 fabs(_Float16 x) { return __ocml_fabs_f16(x); }
__forceinline__ __device__ _Float16 fmin(_Float16 x, _Float16 y) { return __ocml_fmin_f16(x, y); }
__forceinline__ __device__ _Float16 fmax(_Float16 x, _Float16 y) { return __ocml_fmax_f16(x, y); }
__forceinline__ __device__ _Float16 pow(_Float16 x, _Float16 y)
{
    return __ocml_exp_f16(y * __ocml_log_f16(x));
}
__forceinline__ __device__ _Float16 tanh(_Float16 x)
{
    float x_scaled = static_cast<float>(x) * 1.4426950408889634f; // 0x1.715476p+0f = log2(e)
    float a        = __builtin_amdgcn_exp2f(x_scaled);
    float b        = __builtin_amdgcn_exp2f(-x_scaled);

    _Float16 ret = static_cast<_Float16>((a - b) * __builtin_amdgcn_rcpf(a + b));
    _Float16 one = __builtin_copysignf(1.0f, x);

    return __ocml_fabs_f16(x) > 4.5f ? one : ret;
}

__forceinline__ __device__ _Float16 fma(_Float16 a, _Float16 b, _Float16 c)
{
    return __hfma(__half(a), __half(b), __half(c));
}

__forceinline__ __device__ bool isnan(_Float16 a) { return __hisnan(__half(a)); }
__forceinline__ __device__ bool isinf(_Float16 a) { return __hisinf(__half(a)); }

//=============================================================================
// BFloat16 overloads
//=============================================================================

__forceinline__ __device__ ushort exp(ushort x)
{
    return __bfloat16_as_ushort(hexp(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort log(ushort x)
{
    return __bfloat16_as_ushort(hlog(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort sqrt(ushort x)
{
    return __bfloat16_as_ushort(hsqrt(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort rsqrt(ushort x)
{
    return __bfloat16_as_ushort(hrsqrt(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort sin(ushort x)
{
    return __bfloat16_as_ushort(hsin(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort cos(ushort x)
{
    return __bfloat16_as_ushort(hcos(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort fabs(ushort x)
{
    return __bfloat16_as_ushort(__habs(__ushort_as_bfloat16(x)));
}
__forceinline__ __device__ ushort fmax(ushort x, ushort y)
{
    return __bfloat16_as_ushort(__float2bfloat16(fmax(__bfloat162float(__ushort_as_bfloat16(x)),
                                                      __bfloat162float(__ushort_as_bfloat16(y)))));
}
__forceinline__ __device__ ushort fmin(ushort x, ushort y)
{
    return __bfloat16_as_ushort(__float2bfloat16(fmin(__bfloat162float(__ushort_as_bfloat16(x)),
                                                      __bfloat162float(__ushort_as_bfloat16(y)))));
}

__forceinline__ __device__ ushort pow(ushort x, ushort y)
{
    __hip_bfloat16 bf_x = __ushort_as_bfloat16(x);
    __hip_bfloat16 bf_y = __ushort_as_bfloat16(y);
    return __bfloat16_as_ushort(hexp(__hmul(bf_y, hlog(bf_x))));
}
__forceinline__ __device__ ushort tan(ushort x)
{
    __hip_bfloat16 bf_x = __ushort_as_bfloat16(x);
    return __bfloat16_as_ushort(__hdiv(hsin(bf_x), hcos(bf_x)));
}
__forceinline__ __device__ ushort tanh(ushort x)
{
    __hip_bfloat16 bf_x        = __ushort_as_bfloat16(x);
    __hip_bfloat16 two         = __float2bfloat16(2.0f);
    __hip_bfloat16 one         = __float2bfloat16(1.0f);
    __hip_bfloat16 exp2x       = hexp(__hmul(two, bf_x));
    __hip_bfloat16 numerator   = __hsub(exp2x, one);
    __hip_bfloat16 denominator = __hadd(exp2x, one);
    return __bfloat16_as_ushort(__hdiv(numerator, denominator));
}

__forceinline__ __device__ ushort fma(ushort a, ushort b, ushort c)
{
    __hip_bfloat16 bf_a = __ushort_as_bfloat16(a);
    __hip_bfloat16 bf_b = __ushort_as_bfloat16(b);
    __hip_bfloat16 bf_c = __ushort_as_bfloat16(c);

    return __hfma(bf_a, bf_b, bf_c);
}

__forceinline__ __device__ bool isnan(ushort a) { return __hisnan(__ushort_as_bfloat16(a)); }
__forceinline__ __device__ bool isinf(ushort a) { return __hisinf(__ushort_as_bfloat16(a)); }

//=============================================================================
// Double precision overloads
//=============================================================================

__forceinline__ __device__ double exp(double x) { return ::exp(x); }
__forceinline__ __device__ double log(double x) { return ::log(x); }
__forceinline__ __device__ double sqrt(double x) { return ::sqrt(x); }
__forceinline__ __device__ double rsqrt(double x) { return ::rsqrt(x); }
__forceinline__ __device__ double sin(double x) { return ::sin(x); }
__forceinline__ __device__ double cos(double x) { return ::cos(x); }
__forceinline__ __device__ double tan(double x) { return ::tan(x); }
__forceinline__ __device__ double tanh(double x) { return ::tanh(x); }
__forceinline__ __device__ double pow(double x, double y) { return ::pow(x, y); }
__forceinline__ __device__ double fabs(double x) { return ::fabs(x); }
__forceinline__ __device__ double fmax(double x, double y) { return ::fmax(x, y); }
__forceinline__ __device__ double fmin(double x, double y) { return ::fmin(x, y); }
__forceinline__ __device__ double fma(double a, double b, double c) { return ::fma(a, b, c); }
__forceinline__ __device__ bool isnan(double a) { return ::isnan(a); }
__forceinline__ __device__ bool isinf(double a) { return ::isinf(a); }

//=============================================================================
// 4-element vector overloads
//=============================================================================

template <typename FpType> // do std::enable_if?
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
exp(typename mapped_vector_type<FpType, 4>::type x)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = ::exp(x.x);
    out.y                                            = ::exp(x.y);
    out.z                                            = ::exp(x.z);
    out.w                                            = ::exp(x.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
log(typename mapped_vector_type<FpType, 4>::type x)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = log(x.x);
    out.y                                            = log(x.y);
    out.z                                            = log(x.z);
    out.w                                            = log(x.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
sqrt(typename mapped_vector_type<FpType, 4>::type x)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = sqrt(x.x);
    out.y                                            = sqrt(x.y);
    out.z                                            = sqrt(x.z);
    out.w                                            = sqrt(x.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
rsqrt(typename mapped_vector_type<FpType, 4>::type x)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = rsqrt(x.x);
    out.y                                            = rsqrt(x.y);
    out.z                                            = rsqrt(x.z);
    out.w                                            = rsqrt(x.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
fma(typename mapped_vector_type<FpType, 4>::type a,
    typename mapped_vector_type<FpType, 4>::type b,
    typename mapped_vector_type<FpType, 4>::type c)
{
    typename mapped_vector_type<FpType, 4>::type out;
    out.x = fma(a.x, b.x, c.x);
    out.y = fma(a.y, b.y, c.y);
    out.z = fma(a.z, b.z, c.z);
    out.w = fma(a.w, b.w, c.w);

    return out;
}

// __forceinline__ __device__ double sin(double x) { return ::sin(x); }
// __forceinline__ __device__ double cos(double x) { return ::cos(x); }
// __forceinline__ __device__ double tan(double x) { return ::tan(x); }
// __forceinline__ __device__ double tanh(double x) { return ::tanh(x); }
// __forceinline__ __device__ double pow(double x, double y) { return ::pow(x, y); }
// __forceinline__ __device__ double fabs(double x) { return ::fabs(x); }

// TODO: consult with OpenCL spec if that is indeed how these funcs are implemented
template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
fmax(typename mapped_vector_type<FpType, 4>::type x, typename mapped_vector_type<FpType, 4>::type y)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = fmax(x.x, y.x);
    out.y                                            = fmax(x.y, y.y);
    out.z                                            = fmax(x.z, y.z);
    out.w                                            = fmax(x.w, y.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ typename mapped_vector_type<FpType, 4>::type
fmin(typename mapped_vector_type<FpType, 4>::type x, typename mapped_vector_type<FpType, 4>::type y)
{
    typename mapped_vector_type<FpType, 4>::type out = x;
    out.x                                            = fmin(x.x, y.x);
    out.y                                            = fmin(x.y, y.y);
    out.z                                            = fmin(x.z, y.z);
    out.w                                            = fmin(x.w, y.w);
    return out;
}

__forceinline__ __device__ typename mapped_vector_type<float, 4>::type
fma(typename mapped_vector_type<float, 4>::type a,
    typename mapped_vector_type<float, 4>::type b,
    typename mapped_vector_type<float, 4>::type c)
{
    typename mapped_vector_type<float, 4>::type out;
    out.x = fma(a.x, b.x, c.x);
    out.y = fma(a.y, b.y, c.y);
    out.z = fma(a.z, b.z, c.z);
    out.w = fma(a.w, b.w, c.w);

    return out;
}

__forceinline__ __device__ typename mapped_vector_type<float, 4>::type
fmax(typename mapped_vector_type<float, 4>::type x, typename mapped_vector_type<float, 4>::type y)
{
    typename mapped_vector_type<float, 4>::type out = x;
    out.x                                           = fmax(x.x, y.x);
    out.y                                           = fmax(x.y, y.y);
    out.z                                           = fmax(x.z, y.z);
    out.w                                           = fmax(x.w, y.w);
    return out;
}

__forceinline__ __device__ typename mapped_vector_type<float, 4>::type
fmin(typename mapped_vector_type<float, 4>::type x, typename mapped_vector_type<float, 4>::type y)
{
    typename mapped_vector_type<float, 4>::type out = x;
    out.x                                           = fmin(x.x, y.x);
    out.y                                           = fmin(x.y, y.y);
    out.z                                           = fmin(x.z, y.z);
    out.w                                           = fmin(x.w, y.w);
    return out;
}

__forceinline__ __device__ typename mapped_vector_type<float, 4>::type
rsqrt(typename mapped_vector_type<float, 4>::type x)
{
    typename mapped_vector_type<float, 4>::type out = x;
    out.x                                           = rsqrt(x.x);
    out.y                                           = rsqrt(x.y);
    out.z                                           = rsqrt(x.z);
    out.w                                           = rsqrt(x.w);
    return out;
}

template <typename FpType>
__forceinline__ __device__ FpType min(FpType x, FpType y)
{
    return fmin(x, y);
}

template <typename FpType>
__forceinline__ __device__ FpType max(FpType x, FpType y)
{
    return fmax(x, y);
}

} // namespace miopen

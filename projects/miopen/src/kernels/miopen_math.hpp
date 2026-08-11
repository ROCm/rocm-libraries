// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "bfloat16_dev.hpp"
#include "float_types.h"
#include "vector_types.hpp"

namespace miopen {
namespace detail {

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

// The following overloads with __half are needed due to a regression
// in the current implementation of the RNNHiddenStateUpdate kernel

__forceinline__ __device__ __half exp(__half x)
{
    return static_cast<__half>(exp(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half log(__half x)
{
    return static_cast<__half>(log(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half sqrt(__half x)
{
    return static_cast<__half>(sqrt(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half rsqrt(__half x)
{
    return static_cast<__half>(rsqrt(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half sin(__half x)
{
    return static_cast<__half>(sin(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half cos(__half x)
{
    return static_cast<__half>(cos(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half fabs(__half x)
{
    return static_cast<__half>(fabs(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half fmin(__half x)
{
    return static_cast<__half>(fmin(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half fmax(__half x)
{
    return static_cast<__half>(fmax(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half pow(__half x)
{
    return static_cast<__half>(pow(static_cast<_Float16>(x)));
}
__forceinline__ __device__ __half tanh(__half x)
{
    return static_cast<__half>(tanh(static_cast<_Float16>(x)));
}

//=============================================================================
// BFloat16 overloads
//=============================================================================

using bf16_ushort_conversion_t = union
{
    unsigned short int usi;
    __bf16 bf16;
};

__forceinline__ __device__ ushort exp(ushort x)
{
    return float_to_bfloat16(exp(bfloat16_to_float(x)));
}

__forceinline__ __device__ ushort log(ushort x)
{
    return float_to_bfloat16(log(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort sqrt(ushort x)
{
    return float_to_bfloat16(sqrt(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort rsqrt(ushort x)
{
    return float_to_bfloat16(rsqrt(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort sin(ushort x)
{
    return float_to_bfloat16(sin(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort cos(ushort x)
{
    return float_to_bfloat16(cos(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort fabs(ushort x)
{
    return float_to_bfloat16(fabsf(bfloat16_to_float(x)));
}
__forceinline__ __device__ ushort fmax(ushort x, ushort y)
{
    return float_to_bfloat16(fmax(bfloat16_to_float(x), bfloat16_to_float(y)));
}
__forceinline__ __device__ ushort fmin(ushort x, ushort y)
{
    return float_to_bfloat16(fmin(bfloat16_to_float(x), bfloat16_to_float(y)));
}

__forceinline__ __device__ ushort pow(ushort x, ushort y)
{
    bf16_ushort_conversion_t bf_x{x};
    bf16_ushort_conversion_t bf_y{y};

    bf16_ushort_conversion_t bf_mul{.bf16 = bf_x.bf16 * bf_y.bf16};

    return float_to_bfloat16(exp(bfloat16_to_float(bf_mul.usi)));
}
__forceinline__ __device__ ushort tan(ushort x)
{
    bf16_ushort_conversion_t bf_x{x};
    bf16_ushort_conversion_t sinVal{float_to_bfloat16(sin(bfloat16_to_float(bf_x.usi)))};
    bf16_ushort_conversion_t cosVal{float_to_bfloat16(cos(bfloat16_to_float(bf_x.usi)))};

    return bf16_ushort_conversion_t{.bf16 = sinVal.bf16 / cosVal.bf16}.usi;
}
__forceinline__ __device__ ushort tanh(ushort x)
{
    bf16_ushort_conversion_t bf_x{x};
    bf16_ushort_conversion_t two{.usi = float_to_bfloat16(2.0f)};
    bf16_ushort_conversion_t one{.usi = float_to_bfloat16(1.0f)};
    bf16_ushort_conversion_t exp2x{exp(float_to_bfloat16(two.bf16 * bf_x.bf16))};
    bf16_ushort_conversion_t numerator{.bf16 = exp2x.bf16 - one.bf16};
    bf16_ushort_conversion_t denominator{.bf16 = exp2x.bf16 + one.bf16};
    return bf16_ushort_conversion_t{.bf16 = numerator.bf16 / denominator.bf16}.usi;
}

__forceinline__ __device__ ushort fma(ushort a, ushort b, ushort c)
{
    bf16_ushort_conversion_t bf_a{a};
    bf16_ushort_conversion_t bf_b{b};
    bf16_ushort_conversion_t bf_c{c};

    return bf16_ushort_conversion_t{.bf16 =
                                        __builtin_elementwise_fma(bf_a.bf16, bf_b.bf16, bf_c.bf16)}
        .usi;
}

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

} // namespace detail

template <typename FpVecType>
__forceinline__ __device__ FpVecType exp(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::exp(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::exp(inp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType log(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::log(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::log(inp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType sqrt(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::sqrt(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::sqrt(inp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType rsqrt(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::rsqrt(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::rsqrt(inp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType fma(FpVecType a, FpVecType b, FpVecType c)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::fma(a, b, c);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp     = reinterpret_cast<ScalarType*>(&out);
        const auto* ap = reinterpret_cast<const ScalarType*>(&a);
        const auto* bp = reinterpret_cast<const ScalarType*>(&b);
        const auto* cp = reinterpret_cast<const ScalarType*>(&c);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::fma(ap[i], bp[i], cp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType fmax(FpVecType x, FpVecType y)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::fmax(x, y);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp     = reinterpret_cast<ScalarType*>(&out);
        const auto* xp = reinterpret_cast<const ScalarType*>(&x);
        const auto* yp = reinterpret_cast<const ScalarType*>(&y);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::fmax(xp[i], yp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType fmin(FpVecType x, FpVecType y)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::fmin(x, y);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp     = reinterpret_cast<ScalarType*>(&out);
        const auto* xp = reinterpret_cast<const ScalarType*>(&x);
        const auto* yp = reinterpret_cast<const ScalarType*>(&y);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::fmin(xp[i], yp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType min(FpVecType x, FpVecType y)
{
    return fmin(x, y);
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType max(FpVecType x, FpVecType y)
{
    return fmax(x, y);
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType tanh(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::tanh(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::tanh(inp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType pow(FpVecType x, FpVecType y)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::pow(x, y);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp     = reinterpret_cast<ScalarType*>(&out);
        const auto* xp = reinterpret_cast<const ScalarType*>(&x);
        const auto* yp = reinterpret_cast<const ScalarType*>(&y);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::pow(xp[i], yp[i]);
        return out;
    }
}

template <typename FpVecType>
__forceinline__ __device__ FpVecType fabs(FpVecType x)
{
    constexpr auto VecSize = mapped_vector_info<FpVecType>::size;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4 || VecSize == 8,
                  "Unsupported miopen vector operation.");
    if constexpr(VecSize == 1)
    {
        return detail::fabs(x);
    }
    else
    {
        using ScalarType = typename mapped_vector_info<FpVecType>::UnderlyingType;
        FpVecType out;
        auto* outp      = reinterpret_cast<ScalarType*>(&out);
        const auto* inp = reinterpret_cast<const ScalarType*>(&x);
        for(int i = 0; i < static_cast<int>(VecSize); ++i)
            outp[i] = detail::fabs(inp[i]);
        return out;
    }
}

} // namespace miopen

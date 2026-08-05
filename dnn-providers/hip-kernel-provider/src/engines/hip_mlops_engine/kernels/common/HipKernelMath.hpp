// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "VectorTypes.hpp"

namespace hip_kernel_provider
{
namespace detail
{

//=============================================================================
// Float overloads
//=============================================================================

__forceinline__ __device__ float exp(float x)
{
    return expf(x);
}
__forceinline__ __device__ float log(float x)
{
    return logf(x);
}
__forceinline__ __device__ float sqrt(float x)
{
    return sqrtf(x);
}
__forceinline__ __device__ float rsqrt(float x)
{
    return rsqrtf(x);
}
__forceinline__ __device__ float sin(float x)
{
    return sinf(x);
}
__forceinline__ __device__ float cos(float x)
{
    return cosf(x);
}
__forceinline__ __device__ float tan(float x)
{
    return tanf(x);
}
__forceinline__ __device__ float tanh(float x)
{
    return tanhf(x);
}
__forceinline__ __device__ float pow(float x, float y)
{
    return powf(x, y);
}
__forceinline__ __device__ float fabs(float x)
{
    return fabsf(x);
}
__forceinline__ __device__ float fmax(float x, float y)
{
    return fmaxf(x, y);
}
__forceinline__ __device__ float fmin(float x, float y)
{
    return fminf(x, y);
}
__forceinline__ __device__ float fma(float a, float b, float c)
{
    return ::fma(a, b, c);
}

//=============================================================================
// Half precision overloads
//=============================================================================

__forceinline__ __device__ _Float16 exp(_Float16 x)
{
    return __ocml_exp_f16(x);
}
__forceinline__ __device__ _Float16 log(_Float16 x)
{
    return __ocml_log_f16(x);
}
__forceinline__ __device__ _Float16 sqrt(_Float16 x)
{
    return __ocml_sqrt_f16(x);
}
__forceinline__ __device__ _Float16 rsqrt(_Float16 x)
{
    return __ocml_rsqrt_f16(x);
}
__forceinline__ __device__ _Float16 sin(_Float16 x)
{
    return hsin(__half(x));
}
__forceinline__ __device__ _Float16 cos(_Float16 x)
{
    return hcos(__half(x));
}
__forceinline__ __device__ _Float16 fabs(_Float16 x)
{
    return __ocml_fabs_f16(x);
}
__forceinline__ __device__ _Float16 fmin(_Float16 x, _Float16 y)
{
    return __ocml_fmin_f16(x, y);
}
__forceinline__ __device__ _Float16 fmax(_Float16 x, _Float16 y)
{
    return __ocml_fmax_f16(x, y);
}
__forceinline__ __device__ _Float16 pow(_Float16 x, _Float16 y)
{
    return __ocml_exp_f16(y * __ocml_log_f16(x));
}
__forceinline__ __device__ _Float16 tanh(_Float16 x)
{
    float x_scaled = static_cast<float>(x) * 1.4426950408889634f; // 0x1.715476p+0f = log2(e)
    float a = __builtin_amdgcn_exp2f(x_scaled);
    float b = __builtin_amdgcn_exp2f(-x_scaled);

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
__forceinline__ __device__ __half fmin(__half x, __half y)
{
    return static_cast<__half>(fmin(static_cast<_Float16>(x), static_cast<_Float16>(y)));
}
__forceinline__ __device__ __half fmax(__half x, __half y)
{
    return static_cast<__half>(fmax(static_cast<_Float16>(x), static_cast<_Float16>(y)));
}
__forceinline__ __device__ __half pow(__half x, __half y)
{
    return static_cast<__half>(pow(static_cast<_Float16>(x), static_cast<_Float16>(y)));
}
__forceinline__ __device__ __half tanh(__half x)
{
    return static_cast<__half>(tanh(static_cast<_Float16>(x)));
}

//=============================================================================
// BFloat16 overloads
//=============================================================================

__forceinline__ __device__ __bf16 exp(__bf16 x)
{
    return static_cast<__bf16>(exp(static_cast<float>(x)));
}

__forceinline__ __device__ __bf16 log(__bf16 x)
{
    return static_cast<__bf16>(log(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 sqrt(__bf16 x)
{
    return static_cast<__bf16>(sqrt(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 rsqrt(__bf16 x)
{
    return static_cast<__bf16>(rsqrt(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 sin(__bf16 x)
{
    return static_cast<__bf16>(sin(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 cos(__bf16 x)
{
    return static_cast<__bf16>(cos(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 fabs(__bf16 x)
{
    return static_cast<__bf16>(fabsf(static_cast<float>(x)));
}
__forceinline__ __device__ __bf16 fmax(__bf16 x, __bf16 y)
{
    return static_cast<__bf16>(fmax(static_cast<float>(x), static_cast<float>(y)));
}
__forceinline__ __device__ __bf16 fmin(__bf16 x, __bf16 y)
{
    return static_cast<__bf16>(fmin(static_cast<float>(x), static_cast<float>(y)));
}

__forceinline__ __device__ __bf16 pow(__bf16 x, __bf16 y)
{
    return static_cast<__bf16>(pow(static_cast<float>(x), static_cast<float>(y)));
}
__forceinline__ __device__ __bf16 tan(__bf16 x)
{
    float sinVal = sin(static_cast<float>(x));
    float cosVal = cos(static_cast<float>(x));

    return static_cast<__bf16>(sinVal / cosVal);
}
__forceinline__ __device__ __bf16 tanh(__bf16 x)
{
    const auto two = static_cast<__bf16>(2.0f);
    const auto one = static_cast<__bf16>(1.0f);
    __bf16 exp2x = static_cast<__bf16>(exp(static_cast<float>(two * x)));
    __bf16 numerator = exp2x - one;
    __bf16 denominator = exp2x + one;
    return (numerator / denominator);
}

__forceinline__ __device__ __bf16 fma(__bf16 a, __bf16 b, __bf16 c)
{
    return __builtin_elementwise_fma(a, b, c);
}

//=============================================================================
// Double precision overloads
//=============================================================================

__forceinline__ __device__ double exp(double x)
{
    return ::exp(x);
}
__forceinline__ __device__ double log(double x)
{
    return ::log(x);
}
__forceinline__ __device__ double sqrt(double x)
{
    return ::sqrt(x);
}
__forceinline__ __device__ double rsqrt(double x)
{
    return ::rsqrt(x);
}
__forceinline__ __device__ double sin(double x)
{
    return ::sin(x);
}
__forceinline__ __device__ double cos(double x)
{
    return ::cos(x);
}
__forceinline__ __device__ double tan(double x)
{
    return ::tan(x);
}
__forceinline__ __device__ double tanh(double x)
{
    return ::tanh(x);
}
__forceinline__ __device__ double pow(double x, double y)
{
    return ::pow(x, y);
}
__forceinline__ __device__ double fabs(double x)
{
    return ::fabs(x);
}
__forceinline__ __device__ double fmax(double x, double y)
{
    return ::fmax(x, y);
}
__forceinline__ __device__ double fmin(double x, double y)
{
    return ::fmin(x, y);
}
__forceinline__ __device__ double fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

} // namespace detail

//=============================================================================
// 4-element vector overloads
//=============================================================================

#define SINGLE_OPERAND_VEC_MATH(BASE)                                                  \
    template <typename FpVecType>                                                      \
    __forceinline__ __device__ FpVecType BASE(FpVecType x)                             \
    {                                                                                  \
        constexpr auto VecSize = mapped_vector_info<FpVecType>::size;                  \
        if constexpr(VecSize == 4)                                                     \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x);                                                 \
            out.y = detail::BASE(x.y);                                                 \
            out.z = detail::BASE(x.z);                                                 \
            out.w = detail::BASE(x.w);                                                 \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 2)                                                \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x);                                                 \
            out.y = detail::BASE(x.y);                                                 \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 1)                                                \
        {                                                                              \
            return detail::BASE(x);                                                    \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            static_assert(false, "Unsupported hip-kernel-provider vector operation."); \
        }                                                                              \
    }

SINGLE_OPERAND_VEC_MATH(exp);
SINGLE_OPERAND_VEC_MATH(log);
SINGLE_OPERAND_VEC_MATH(sqrt);
SINGLE_OPERAND_VEC_MATH(rsqrt);
SINGLE_OPERAND_VEC_MATH(tanh);
SINGLE_OPERAND_VEC_MATH(fabs);

#undef SINGLE_OPERAND_VEC_MATH

#define DUAL_OPERAND_VEC_MATH(BASE)                                                    \
    template <typename FpVecType>                                                      \
    __forceinline__ __device__ FpVecType BASE(FpVecType x, FpVecType y)                \
    {                                                                                  \
        constexpr auto VecSize = mapped_vector_info<FpVecType>::size;                  \
        if constexpr(VecSize == 4)                                                     \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x, y.x);                                            \
            out.y = detail::BASE(x.y, y.y);                                            \
            out.z = detail::BASE(x.z, y.z);                                            \
            out.w = detail::BASE(x.w, y.w);                                            \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 2)                                                \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x, y.x);                                            \
            out.y = detail::BASE(x.y, y.y);                                            \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 1)                                                \
        {                                                                              \
            return detail::BASE(x, y);                                                 \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            static_assert(false, "Unsupported hip-kernel-provider vector operation."); \
        }                                                                              \
    }

DUAL_OPERAND_VEC_MATH(fmin);
DUAL_OPERAND_VEC_MATH(fmax);
DUAL_OPERAND_VEC_MATH(pow);

// Forward calls from hip_kernel_provider::min() to fmin
template <typename FpVecType>
__forceinline__ __device__ FpVecType min(FpVecType x, FpVecType y)
{
    return fmin(x, y);
}

// Forward calls from hip_kernel_provider::max() to fmax
template <typename FpVecType>
__forceinline__ __device__ FpVecType max(FpVecType x, FpVecType y)
{
    return fmax(x, y);
}

#undef DUAL_OPERAND_VEC_MATH

#define TRIPLE_OPERAND_VEC_MATH(BASE)                                                  \
    template <typename FpVecType>                                                      \
    __forceinline__ __device__ FpVecType BASE(FpVecType x, FpVecType y, FpVecType z)   \
    {                                                                                  \
        constexpr auto VecSize = mapped_vector_info<FpVecType>::size;                  \
        if constexpr(VecSize == 4)                                                     \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x, y.x, z.x);                                       \
            out.y = detail::BASE(x.y, y.y, z.y);                                       \
            out.z = detail::BASE(x.z, y.z, z.z);                                       \
            out.w = detail::BASE(x.w, y.w, z.w);                                       \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 2)                                                \
        {                                                                              \
            FpVecType out;                                                             \
            out.x = detail::BASE(x.x, y.x, z.x);                                       \
            out.y = detail::BASE(x.y, y.y, z.y);                                       \
            return out;                                                                \
        }                                                                              \
        else if constexpr(VecSize == 1)                                                \
        {                                                                              \
            return detail::BASE(x, y, z);                                              \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            static_assert(false, "Unsupported hip-kernel-provider vector operation."); \
        }                                                                              \
    }

TRIPLE_OPERAND_VEC_MATH(fma);

#undef TRIPLE_OPERAND_VEC_MATH

} // namespace hip_kernel_provider

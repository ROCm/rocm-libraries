// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_FMA_FROM_TENSOR_HPP
#define CK_AMD_FMA_FROM_TENSOR_HPP

#include "data_type.hpp"
namespace ck {

// aux_data[ 5: 0] maps to instr[ 63: 58] for VOP2M-VOP6M (MOD0)
// aux_data[17: 6] maps to instr[ 87: 76] for VOP3M-VOP5M (MOD1)
// aux_data[13: 6] maps to instr[ 83: 76] for VOP6M       (MOD1)
// aux_data[25:18] maps to instr[127:120] for VOP5M-VOP6M (MOD2)
// constexpr index_t FmaAuxData_Mod0 = 0;
constexpr index_t FmaAuxData_Mod1 = 6;
// constexpr index_t FmaAuxData_Mod2 = 18;

template <index_t ChanOff>
constexpr index_t GetChanOff()
{
    // 0->0, 2->1, 8->2, 10->3, 16->4, 18->5
    return ((ChanOff + 2) / 4) << (FmaAuxData_Mod1 + 4);
};

template <typename Y,
          typename X,
          typename enable_if<sizeof(X) == 2 * sizeof(Y), bool>::type = false>
__host__ __device__ constexpr void SplitVector(const X& x, Y& low, Y& high)
{
    using TupleVec2 = Tuple<Y, Y>;
    low             = bit_cast<TupleVec2>(x)[Number<0>{}];
    high            = bit_cast<TupleVec2>(x)[Number<1>{}];
}

template <index_t H, index_t W, index_t ChanOff>
struct intrin_wcnn_fma_from_tensor;

template <index_t ChanOff>
struct intrin_wcnn_fma_from_tensor<4, 2, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const int4x8_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr bool clamp        = false;
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 2>();
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i4_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i4_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const int4x8_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr bool clamp       = false;
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const int4x8_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const uint4x8_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 2>();
        constexpr bool clamp        = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u4_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u4_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const uint4x8_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const uint4x8_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }
#endif
    // i8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const int8x4_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = 0;
        constexpr index_t aux_data1 = GetChanOff<2>();
        constexpr bool clamp        = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i8_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i8_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const int8x4_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const int8x4_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const uint8x4_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = 0;
        constexpr index_t aux_data1 = GetChanOff<2>();
        constexpr bool clamp        = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u8_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u8_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const uint8x4_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const uint8x4_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const f8x4_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = 0;
        constexpr index_t aux_data1 = GetChanOff<2>();
        constexpr bool clamp        = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_fp8_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_fp8_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const f8x4_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const f8x4_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const bf8x4_t& residual0, const float4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = 0;
        constexpr index_t aux_data1 = GetChanOff<2>();
        constexpr bool clamp        = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf8_4x2(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf8_4x2(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const bf8x4_t& residual0, const half4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const bf8x4_t& residual0, const bhalf4_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_4x2(
                inAcc, bit_cast<uint32_t>(residual0), scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const float4_t& inAcc,
                               const half2_t& residual0,
                               const half2_t& residual1,
                               const float4_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_f16_4x2(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_f16_4x2(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const half4_t& inAcc,
                               const half2_t& residual0,
                               const half2_t& residual1,
                               const half4_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x2(
                inAcc, residual0, residual1, scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const float4_t& inAcc,
                               const bhalf2_t& residual0,
                               const bhalf2_t& residual1,
                               const float4_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        float2_t inAcc0, inAcc1;
        float2_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<float2_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf16_4x2(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<float2_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf16_4x2(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf4_t& inAcc,
                               const bhalf2_t& residual0,
                               const bhalf2_t& residual1,
                               const bhalf4_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x2(
                inAcc, residual0, residual1, scale, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }
};

template <index_t ChanOff>
struct intrin_wcnn_fma_from_tensor<4, 4, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int4x8_t& residual0, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 8>();
        constexpr bool clamp        = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_4x4(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int4x8_t& residual0, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 8>();
        constexpr bool clamp        = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_4x4(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint4x8_t& residual0, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 8>();
        constexpr bool clamp        = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_4x4(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint4x8_t& residual0, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff + 8>();
        constexpr bool clamp        = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data0, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_4x4(
                inAcc1, bit_cast<uint32_t>(residual0), scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = scale;
        ignore = outAcc;
#endif
    }
#endif
    // i8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const int8x4_t& residual0,
                               const int8x4_t& residual1,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const int8x4_t& residual0,
                               const int8x4_t& residual1,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const uint8x4_t& residual0,
                               const uint8x4_t& residual1,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const uint8x4_t& residual0,
                               const uint8x4_t& residual1,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const f8x4_t& residual0,
                               const f8x4_t& residual1,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const f8x4_t& residual0,
                               const f8x4_t& residual1,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const bf8x4_t& residual0,
                               const bf8x4_t& residual1,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bf8x4_t& residual0,
                               const bf8x4_t& residual1,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_4x4(
                inAcc0, bit_cast<uint32_t>(residual0), scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_4x4(
                inAcc1, bit_cast<uint32_t>(residual1), scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const half2_t& residual0,
                               const half2_t& residual1,
                               const half2_t& residual2,
                               const half2_t& residual3,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x4(
                inAcc0, residual0, residual1, scale0, aux_data, clamp);

        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x4(
                inAcc1, residual2, residual3, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = residual2;
        ignore = residual3;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bhalf2_t& residual0,
                               const bhalf2_t& residual1,
                               const bhalf2_t& residual2,
                               const bhalf2_t& residual3,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x4(
                inAcc0, residual0, residual1, scale0, aux_data, clamp);

        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x4(
                inAcc1, residual2, residual3, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual0;
        ignore = residual1;
        ignore = residual2;
        ignore = residual3;
        ignore = scale;
        ignore = outAcc;
#endif
    }
};

template <index_t ChanOff>
struct intrin_wcnn_fma_from_tensor<8, 4, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int4x16_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff>();
        constexpr bool clamp        = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_8x4(
                inAcc0, residual0, scale0, aux_data0, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_8x4(
                inAcc1, residual1, scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int4x16_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff>();
        constexpr bool clamp        = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_8x4(
                inAcc0, residual0, scale0, aux_data0, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_8x4(
                inAcc1, residual1, scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint4x16_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff>();
        constexpr bool clamp        = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_8x4(
                inAcc0, residual0, scale0, aux_data0, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_8x4(
                inAcc1, residual1, scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint4x16_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data0 = GetChanOff<ChanOff>();
        constexpr index_t aux_data1 = GetChanOff<ChanOff>();
        constexpr bool clamp        = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_8x4(
                inAcc0, residual0, scale0, aux_data0, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_8x4(
                inAcc1, residual1, scale1, aux_data1, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }
#endif

    // i8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int8x8_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_8x4(inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_8x4(inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int8x8_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_8x4(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // u8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint8x8_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_8x4(inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_8x4(inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint8x8_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_8x4(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const f8x8_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_8x4(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const f8x8_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_8x4(
                inAcc0, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const bf8x8_t& residual, const half8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_8x4(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const bf8x8_t& residual, const bhalf8_t scale, FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        const int32_t residual0 = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_8x4(
                inAcc0, residual0, scale0, aux_data, clamp);
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_8x4(
                inAcc1, residual1, scale1, aux_data, clamp);
#else
        ignore = inAcc;
        ignore = residual;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const half4_t& residual01,
                               const half4_t& residual23,
                               const half8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        // c0-c7
        const int32_t residual0 = bit_cast<int32x2_t>(residual01)[0];
        const int32_t residual1 = bit_cast<int32x2_t>(residual01)[1];
        // c8-c15
        const int32_t residual2 = bit_cast<int32x2_t>(residual23)[0];
        const int32_t residual3 = bit_cast<int32x2_t>(residual23)[1];

        half4_t inAcc0, inAcc1;
        half4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        // first 4x4 tile
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_8x4(inAcc0,
                                                         bit_cast<half2_t>(residual0),
                                                         bit_cast<half2_t>(residual2),
                                                         scale0,
                                                         aux_data,
                                                         clamp);

        // second 4x4 tile
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_8x4(inAcc1,
                                                         bit_cast<half2_t>(residual1),
                                                         bit_cast<half2_t>(residual3),
                                                         scale1,
                                                         aux_data,
                                                         clamp);
#else
        ignore = inAcc;
        ignore = residual01;
        ignore = residual23;
        ignore = scale;
        ignore = outAcc;
#endif
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bhalf4_t& residual01,
                               const bhalf4_t& residual23,
                               const bhalf8_t scale,
                               FloatAcc& outAcc)
    {
#if defined(__gfx13__)
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual0    = bit_cast<int32x2_t>(residual01)[0];
        const int32_t residual1    = bit_cast<int32x2_t>(residual01)[1];
        const int32_t residual2    = bit_cast<int32x2_t>(residual23)[0];
        const int32_t residual3    = bit_cast<int32x2_t>(residual23)[1];

        bhalf4_t inAcc0, inAcc1;
        bhalf4_t scale0, scale1;
        SplitVector(inAcc, inAcc0, inAcc1);
        SplitVector(scale, scale0, scale1);
        // first 4x4 tile
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_8x4(inAcc0,
                                                           bit_cast<bhalf2_t>(residual0),
                                                           bit_cast<bhalf2_t>(residual2),
                                                           scale0,
                                                           aux_data,
                                                           clamp);
        // second 4x4 tile
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_8x4(inAcc1,
                                                           bit_cast<bhalf2_t>(residual1),
                                                           bit_cast<bhalf2_t>(residual3),
                                                           scale1,
                                                           aux_data,
                                                           clamp);
#else
        ignore = inAcc;
        ignore = residual01;
        ignore = residual23;
        ignore = scale;
        ignore = outAcc;
#endif
    }
};

} // namespace ck
#endif // CK_AMD_FMA_FROM_TENSOR_HPP

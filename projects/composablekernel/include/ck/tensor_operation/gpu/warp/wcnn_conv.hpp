// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_conv.hpp"

namespace ck {

enum struct WcnnConvInstr
{
    // gfx13
    wcnn_conv_f32_f16 = 0,
    wcnn_conv_f32_bf16,
    wcnn_conv_f32_f8,
    wcnn_conv_f32_bf8,
    wcnn_conv_f32_iu8,
    wcnn_conv_i32_iu8,
    wcnn_conv_f16_f16,
    wcnn_conv_bf16_bf16,
    wcnn_conv_f16_f8,
    wcnn_conv_f16_bf8,
    wcnn_conv_f16_iu8,
    wcnn_conv_f32_iu4,
    wcnn_conv_i32_iu4,
    wcnn_conv_f16_iu4,
    wcnn_conv_f32i32_iu4,
    wcnn_conv_f32i32_iu8,
    wcnn_conv_f32_f8_bf8,
    wcnn_conv_f32_bf8_f8,
    wcnn_conv_f16_f8_bf8,
    wcnn_conv_f16_bf8_f8,
    wcnn_conv_bf16_iu4, // invalid
};

template <WcnnConvInstr Instr,
          index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false>
struct WcnnConvType
{
    WcnnConvType() { static_assert(false, "never called"); }
};

// dst: f32 or i32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_f16,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_f16<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_bf16,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_bf16<H,
                                  W,
                                  FilterSize,
                                  DilationX,
                                  DilationY,
                                  Iters,
                                  Aco,
                                  Signed,
                                  isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_f8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_f8<H,
                                W,
                                FilterSize,
                                DilationX,
                                DilationY,
                                Iters,
                                Aco,
                                Signed,
                                isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_bf8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_bf8<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_f8_bf8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_f8_bf8<H,
                                    W,
                                    FilterSize,
                                    DilationX,
                                    DilationY,
                                    Iters,
                                    Aco,
                                    Signed,
                                    isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_bf8_f8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_bf8_f8<H,
                                    W,
                                    FilterSize,
                                    DilationX,
                                    DilationY,
                                    Iters,
                                    Aco,
                                    Signed,
                                    isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_iu8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_iu8<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_i32_iu8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_i32_iu8<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32i32_iu8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        constexpr bool isLastPass = (Mod{} & 2) != 0;
        if constexpr(isLastPass)
        {
            intrin_wcnn_conv_f32i32_iu8<H,
                                        W,
                                        FilterSize,
                                        DilationX,
                                        DilationY,
                                        Iters,
                                        Aco,
                                        Signed,
                                        isHighLane>::Run(reg_wei, reg_data, reg_c);
        }
        else
        {
            vector_type<int32_t, 4> tmp_c;
            tmp_c.template AsType<int32x4_t>()(Number<0>{}) =
                bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()(Number<0>{}));
            intrin_wcnn_conv_i32_iu8<H,
                                     W,
                                     FilterSize,
                                     DilationX,
                                     DilationY,
                                     Iters,
                                     Aco,
                                     Signed,
                                     isHighLane>::Run(reg_wei, reg_data, tmp_c);
            reg_c.template AsType<float4_t>()(Number<0>{}) =
                bit_cast<float4_t>(tmp_c.template AsType<int32x4_t>()(Number<0>{}));
        }
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32_iu4,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f32_iu4<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_i32_iu4,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_i32_iu4<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f32i32_iu4,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        constexpr bool isLastPass = (Mod{} & 2) != 0;
        if constexpr(isLastPass)
        {
            intrin_wcnn_conv_f32i32_iu4<H,
                                        W,
                                        FilterSize,
                                        DilationX,
                                        DilationY,
                                        Iters,
                                        Aco,
                                        Signed,
                                        isHighLane>::Run(reg_wei, reg_data, reg_c);
        }
        else
        {
            vector_type<int32_t, 4> tmp_c;
            tmp_c.template AsType<int32x4_t>()(Number<0>{}) =
                bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()(Number<0>{}));
            intrin_wcnn_conv_i32_iu4<H,
                                     W,
                                     FilterSize,
                                     DilationX,
                                     DilationY,
                                     Iters,
                                     Aco,
                                     Signed,
                                     isHighLane>::Run(reg_wei, reg_data, tmp_c);
            reg_c.template AsType<float4_t>()(Number<0>{}) =
                bit_cast<float4_t>(tmp_c.template AsType<int32x4_t>()(Number<0>{}));
        }
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_iu4,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_iu4<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_bf16_iu4,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
        // TODO
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
    }
};
#endif

// dst: f16 or bf16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_f16,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_f16<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_bf16_bf16,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_bf16_bf16<H,
                                   W,
                                   FilterSize,
                                   DilationX,
                                   DilationY,
                                   Iters,
                                   Aco,
                                   Signed,
                                   isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_f8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_f8<H,
                                W,
                                FilterSize,
                                DilationX,
                                DilationY,
                                Iters,
                                Aco,
                                Signed,
                                isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_bf8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_bf8<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_f8_bf8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_f8_bf8<H,
                                    W,
                                    FilterSize,
                                    DilationX,
                                    DilationY,
                                    Iters,
                                    Aco,
                                    Signed,
                                    isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_bf8_f8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_bf8_f8<H,
                                    W,
                                    FilterSize,
                                    DilationX,
                                    DilationY,
                                    Iters,
                                    Aco,
                                    Signed,
                                    isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct WcnnConvType<WcnnConvInstr::wcnn_conv_f16_iu8,
                    H,
                    W,
                    FilterSize,
                    DilationX,
                    DilationY,
                    Iters,
                    Aco,
                    Signed>
{
    template <class FloatA, class FloatB, class FloatC, class Mod>
    __device__ void Run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c, Mod) const
    {
#if defined(__gfx13__)
        constexpr bool isHighLane = Mod{} & 1;
        intrin_wcnn_conv_f16_iu8<H,
                                 W,
                                 FilterSize,
                                 DilationX,
                                 DilationY,
                                 Iters,
                                 Aco,
                                 Signed,
                                 isHighLane>::Run(reg_wei, reg_data, reg_c);
#else
        ignore = reg_wei;
        ignore = reg_data;
        ignore = reg_c;
#endif
    }
};

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool UseF32I32>
struct WcnnConvSelector
{
    template <typename WeiDataType_, typename InDataType_, typename AccDataType_>
    static constexpr auto GetWcnnConv();

    template <>
    constexpr auto GetWcnnConv<half_t, half_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_f16;
    }
    template <>
    constexpr auto GetWcnnConv<bhalf_t, bhalf_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_bf16;
    }

    template <>
    constexpr auto GetWcnnConv<f8_t, f8_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_f8;
    }

    template <>
    constexpr auto GetWcnnConv<bf8_t, bf8_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_bf8;
    }

    template <>
    constexpr auto GetWcnnConv<f8_t, bf8_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_f8_bf8;
    }

    template <>
    constexpr auto GetWcnnConv<bf8_t, f8_t, float_t>()
    {
        return WcnnConvInstr::wcnn_conv_f32_bf8_f8;
    }

    template <>
    constexpr auto GetWcnnConv<int8_t, int8_t, float_t>()
    {
        return UseF32I32 ? WcnnConvInstr::wcnn_conv_f32i32_iu8 : WcnnConvInstr::wcnn_conv_f32_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<uint8_t, uint8_t, float_t>()
    {
        return UseF32I32 ? WcnnConvInstr::wcnn_conv_f32i32_iu8 : WcnnConvInstr::wcnn_conv_f32_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<int4_t, int4_t, float_t>()
    {
        return UseF32I32 ? WcnnConvInstr::wcnn_conv_f32i32_iu4 : WcnnConvInstr::wcnn_conv_f32_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<uint4_t, uint4_t, float_t>()
    {
        return UseF32I32 ? WcnnConvInstr::wcnn_conv_f32i32_iu4 : WcnnConvInstr::wcnn_conv_f32_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<int8_t, int8_t, int32_t>()
    {
        return WcnnConvInstr::wcnn_conv_i32_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<uint8_t, uint8_t, int32_t>()
    {
        return WcnnConvInstr::wcnn_conv_i32_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<int4_t, int4_t, int32_t>()
    {
        return WcnnConvInstr::wcnn_conv_i32_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<uint4_t, uint4_t, int32_t>()
    {
        return WcnnConvInstr::wcnn_conv_i32_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<half_t, half_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_f16;
    }

    template <>
    constexpr auto GetWcnnConv<bhalf_t, bhalf_t, bhalf_t>()
    {
        return WcnnConvInstr::wcnn_conv_bf16_bf16;
    }

    template <>
    constexpr auto GetWcnnConv<f8_t, f8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_f8;
    }

    template <>
    constexpr auto GetWcnnConv<bf8_t, bf8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_bf8;
    }

    template <>
    constexpr auto GetWcnnConv<f8_t, bf8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_f8_bf8;
    }

    template <>
    constexpr auto GetWcnnConv<bf8_t, f8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_bf8_f8;
    }

    template <>
    constexpr auto GetWcnnConv<int8_t, int8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<uint8_t, uint8_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_iu8;
    }

    template <>
    constexpr auto GetWcnnConv<int4_t, int4_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<uint4_t, uint4_t, half_t>()
    {
        return WcnnConvInstr::wcnn_conv_f16_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<int4_t, int4_t, bhalf_t>()
    {
        return WcnnConvInstr::wcnn_conv_bf16_iu4;
    }

    template <>
    constexpr auto GetWcnnConv<uint4_t, uint4_t, bhalf_t>()
    {
        return WcnnConvInstr::wcnn_conv_bf16_iu4;
    }

    static constexpr auto Signed()
    {
        return std::is_same<InDataType, int4_t>::value || std::is_same<InDataType, int8_t>::value;
    }

    static constexpr auto selected_conv =
        WcnnConvType<GetWcnnConv<WeiDataType, InDataType, AccDataType>(),
                     HPerWcnn,
                     WPerWcnn,
                     FilterSize,
                     DilationX,
                     DilationY,
                     Iters,
                     Aco,
                     Signed()>{};

    __host__ __device__ constexpr WcnnConvSelector()
    {
        static_assert(FilterSize == 1 || FilterSize == 3, "WRONG! FilterSize must be 1 or 3");
    };
};

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters  = 1,
          bool WaveGroup = false,
          bool Aco       = false,
          bool UseF32I32 = false>
struct WcnnConv
{
    static constexpr index_t WaveSize = 32;

    __host__ __device__ constexpr WcnnConv(){};

    template <typename ScalarType>
    static constexpr index_t SizeOfBits()
    {
        return std::is_same<int4_t, ScalarType>::value ? 4 : sizeof(ScalarType) * 8;
    }

    static __device__ index_t GetLaneId()
    {
        if constexpr(WaveGroup == false)
        {
            return get_thread_local_1d_id() & (WaveSize - 1);
        }
        else
        {
            return get_lane_id();
        }
    }

    // WCNN input channel and output channel info
    static constexpr index_t GetNumInputChannels()
    {
        return (HPerWcnn == 4) && (WPerWcnn == 2) ? 128 / SizeOfBits<KernelInDataType>()
                                                  : 64 / SizeOfBits<KernelInDataType>();
    }

    static constexpr index_t GetNumOutputChannels()
    {
        return (HPerWcnn == 8) && (WPerWcnn == 4) ? 8 : 16;
    }

    static constexpr index_t GetUnpackedNumInputChannels()
    {
        return (HPerWcnn == 4) && (WPerWcnn == 2) ? 128 / SizeOfBits<InDataType>()
                                                  : 64 / SizeOfBits<InDataType>();
    }

    static constexpr index_t GetUnpackedNumOutputChannels() { return GetNumOutputChannels(); }

    // WCNN weight info
    static constexpr index_t GetWeightRegSize()
    {
        constexpr index_t WeightBitsPerLane = GetNumInputChannels() * GetNumOutputChannels() *
                                              SizeOfBits<KernelWeightDataType>() / WaveSize *
                                              FilterSize * FilterSize * Iters;

        // Round bits to Reg size (uint32_t)
        return (WeightBitsPerLane + 31) / 32;
    }

    static constexpr index_t GetNumWeightComponents()
    {
        return GetWeightRegSize() * 32 / SizeOfBits<KernelWeightDataType>();
    }

    static constexpr index_t GetNumSubTilesPerWeightTap()
    {
        return ((HPerWcnn == 4) && (WPerWcnn == 2)) ? 2 : 1;
    }

    static constexpr index_t GetNumWeightCompPerTile()
    {
        return GetNumWeightComponents() / GetWeightRegSize();
    }

    static constexpr index_t GetNumWeightCompPerTap()
    {
        return GetNumSubTilesPerWeightTap() * GetNumWeightCompPerTile();
    }

    static constexpr index_t GetNumWeightTap()
    {
        if constexpr(FilterSize == 1)
        {
            return 1;
        }
        else if constexpr(FilterSize == 3)
        {
            return (HPerWcnn == 8) && (WPerWcnn == 4) ? 5 : 9;
        }
    }

    static constexpr index_t GetNumWeightTapPerWave()
    {
        return (HPerWcnn == 8) && (WPerWcnn == 4) ? 2 : 1;
    }

    // GFX13 3X3 Filter tap order
    //    -------------
    //    | 2 | 1 | 8 |
    //    -------------
    //    | 3 | 0 | 7 |
    //    -------------
    //    | 4 | 5 | 6 |
    //    -------------
    // The offset of remap table should be applied to the destination of weight data transfer.
    __host__ __device__ static constexpr auto GetWeight3RemapTable()
    {
        return Sequence<2, 1, 8, 3, 0, 7, 4, 5, 6>{};
    }

    __host__ __device__ static constexpr auto GetWeightRemapTable()
    {
        if constexpr(FilterSize == 1)
        {
            return Sequence<0>{};
        }
        else if constexpr(FilterSize == 3)
        {
            if constexpr(GetNumWeightTap() == 9)
            {
                return GetWeight3RemapTable();
            }
            else
            {
                // When pixel shape is 8X4, each VGPR contains two taps. we only need include the
                // even elements with half value from original table. i.e.
                // tap[i] = tap_orig[2*i] / 2
                return Sequence<1, 4, 0, 2, 3>{};
            }
        }
        else
        {
            static_assert(false, "never called");
        }
    }

    // Get additional tap order adjustment for second tap in source.
    // When pixel shape is 8X4, each VGPR contains two taps, and these two taps are not continuous.
    // When transfer 3x3 filter from global to LDS, we needn't consider it.
    // when transfer 3x3 filter from global to VGPR, we have to use this function to adjust the src
    // tap offset in lane 16-31.
    __host__ __device__ static constexpr auto GetWeightSecondTapMapTable()
    {
        if constexpr(FilterSize == 1)
        {
            return Sequence<0>{};
        }
        else if constexpr(FilterSize == 3)
        {
            if constexpr(GetNumWeightTap() == 9)
            {
                return Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{};
            }
            else
            {
                // original order:   0,    1,    2,    3,        4,    5,    6,    7,    8
                // vgpr order:       1_lo, 0_hi, 4_lo, 1_hi,     0_lo, 3_hi, 2_lo, 2_hi, 3_lo
                // adjusted order:   0,    3,    2,    2,        4,    1,    6,    7,    8,    5
                // final vgpr order: 1_lo, 1_hi, 4_lo, (unused), 0_lo, 0_hi, 2_lo, 2_hi, 3_lo, 3_hi
                // map table value = odd tap - even tap in adjusted table
                return Sequence<3, 0, -3, 1, -3>{};
            }
        }
        else
        {
            static_assert(false, "never called");
        }
    }

    // WCNN data info
    static constexpr index_t GetDataRegSizePerTile()
    {
        return HPerWcnn * WPerWcnn * GetNumInputChannels() * SizeOfBits<KernelInDataType>() /
               WaveSize / 32;
    }

    static constexpr index_t GetNumDataCompPerTile()
    {
        return GetNumDataComponents() / GetNumImageSubTilesInVertical();
    }

    static constexpr index_t GetNumSubTilesPerImageTile()
    {
        return ((HPerWcnn == 8) && (WPerWcnn == 4)) ? 2 : 1;
    }

    static constexpr index_t GetNumImageTilesInVertical()
    {
        // Filter 1x1: always require 1 image tile.
        // Filter 3x3: 8x4 require 2 tiles in vertical, other pixel shapes require 3 tiles.
        return FilterSize == 1 ? 1 : ((HPerWcnn == 8) && (WPerWcnn == 4)) ? 2 : 3;
    }

    static constexpr index_t GetNumImageSubTilesInVertical()
    {
        return GetNumImageTilesInVertical() * GetNumSubTilesPerImageTile();
    }

    static constexpr index_t GetDataRegSize()
    {
        return GetDataRegSizePerTile() * GetNumImageTilesInVertical();
    }

    static constexpr index_t GetNumDataComponents()
    {
        return GetDataRegSize() * 32 / SizeOfBits<KernelInDataType>();
    }

    // Accum info
    static constexpr index_t GetAccumRegSize()
    {
        return HPerWcnn * WPerWcnn * GetNumOutputChannels() * SizeOfBits<AccDataType>() / WaveSize /
               32;
    }

    static constexpr index_t GetNumAccumComponents()
    {
        return GetAccumRegSize() * 32 / SizeOfBits<AccDataType>();
    }

    static constexpr index_t GetNumOutTensorComponents()
    {
        return GetAccumRegSize() * 32 / SizeOfBits<AccDataType>();
    }

    static constexpr index_t GetNumOutTensorComponentsFor4bit() { return HPerWcnn / 4; }

    // Helper functions of origin data index per thread
    // {tap_offset x k1 x c1 x c2}
    template <bool TileAccess>
    static constexpr auto CalculateWeiDataThreadOriginDataIndex()
    {
        if constexpr(TileAccess)
        {
            auto laneId = GetLaneId();
            if constexpr((HPerWcnn == 8) && (WPerWcnn == 4))
            {
                return make_tuple(laneId / 8, laneId & (GetNumOutputChannels() - 1), 0, 0);
            }
            else
            {
                return make_tuple(0, laneId, 0, 0);
            }
        }
        else
        {
            auto laneId                      = GetLaneId();
            constexpr index_t NumCompPerTile = GetNumWeightCompPerTile();
            if constexpr((HPerWcnn == 8) && (WPerWcnn == 4))
            {
                return make_tuple(laneId / 16,
                                  (laneId / 2) & (GetNumOutputChannels() - 1),
                                  0,
                                  (laneId % 2) * NumCompPerTile);
            }
            else
            {
                return make_tuple(0, (laneId / 2), 0, (laneId % 2) * NumCompPerTile);
            }
        }
    }

    // {h1 x w1 x c1}
    template <bool TileAccess>
    static constexpr auto CalculateInDataThreadOriginDataIndex()
    {
        if constexpr(TileAccess)
        {

            auto laneId                          = GetLaneId();
            constexpr index_t NumSubTilePerImage = GetNumSubTilesPerImageTile();
            const index_t PixelOffset = laneId % (HPerWcnn * WPerWcnn / NumSubTilePerImage);
            return make_tuple(PixelOffset / WPerWcnn, PixelOffset % WPerWcnn, 0);
        }
        else
        {
            auto laneId                      = GetLaneId();
            constexpr index_t NumCompPerTile = GetNumDataCompPerTile();
            const index_t SubC               = (laneId * NumCompPerTile) % GetNumInputChannels();
            const index_t PixelOffset        = laneId * NumCompPerTile / GetNumInputChannels();

            return make_tuple(PixelOffset / WPerWcnn, PixelOffset % WPerWcnn, SubC);
        }
    }

    template <bool TileAccess>
    static constexpr auto CalculateAccThreadOriginDataIndex()
    {
        auto laneId = GetLaneId();
        if constexpr(TileAccess)
        {
            // {h1 x h2 x w1 x k1}
            const index_t subW = laneId % WPerWcnn;
            const index_t subH = laneId / WPerWcnn;
            return make_tuple(0, subH, subW, 0);
        }
        else
        {
            // {h1 x h2 x w1 x k1 x k2}
            const index_t accCompIdx =
                laneId * GetNumAccumComponents() / GetNumSubTilesPerImageTile();
            const index_t subW = (accCompIdx / GetNumOutputChannels()) % WPerWcnn;
            const index_t subH = (accCompIdx / GetNumOutputChannels()) / WPerWcnn;

            if constexpr(Aco == 0)
            {
                constexpr index_t SwizzleComp = 4;
                const index_t subK            = accCompIdx % GetNumOutputChannels();
                const index_t subK_8          = (laneId & 1) * SwizzleComp;

                if constexpr(GetNumAccumComponents() == 4)
                {
                    return make_tuple(0, subH, subW, 0, subK);
                }
                else
                {
                    return make_tuple(0, subH, subW, 0, subK_8);
                }
            }
            else
            {
                constexpr index_t SwizzleComp    = 2;
                constexpr index_t NumLanePerPair = (WPerWcnn == 2) && (HPerWcnn == 4) ? 4 : 2;
                const index_t subK               = (laneId & (NumLanePerPair - 1)) * SwizzleComp;
                return make_tuple(0, subH, subW, 0, subK);
            }
        }
    }

    template <typename DataType_>
    static auto GetKernelDataType()
    {
        if constexpr(std::is_same<int4_t, DataType_>::value)
        {
            return int32_t();
        }
        else if constexpr(std::is_same<uint4_t, DataType_>::value)
        {
            return uint32_t();
        }
        else
        {
            return DataType_();
        }
    }

    // Pre-defined types
    using KernelWeightDataType = decltype(GetKernelDataType<WeiDataType>());
    using KernelInDataType     = decltype(GetKernelDataType<InDataType>());

    using AccDataVec           = vector_type<AccDataType, GetNumAccumComponents()>;
    using WeiDataVec           = vector_type<KernelWeightDataType, GetNumWeightComponents()>;
    using InDataVec            = vector_type<KernelInDataType, GetNumDataComponents()>;
    using outTensorDataVec     = vector_type<KernelInDataType, GetNumOutTensorComponents()>;
    using out4bitTensorDataVec = vector_type<KernelInDataType, GetNumOutTensorComponentsFor4bit()>;

    using AccDataTileVec =
        vector_type<AccDataType, GetNumAccumComponents() / GetNumSubTilesPerImageTile()>;
    using WeiDataTileVec = vector_type<KernelWeightDataType, GetNumWeightCompPerTile()>;
    using InDataTileVec  = vector_type<KernelInDataType, GetNumDataCompPerTile()>;

    using WeiDataTapVec = vector_type<KernelWeightDataType, GetNumWeightCompPerTap()>;

    static_assert(GetNumImageSubTilesInVertical() == GetDataRegSize(), "");

    // WCNN intrinsic
    static constexpr auto wcnn_conv_selector = WcnnConvSelector<WeiDataType,
                                                                InDataType,
                                                                AccDataType,
                                                                HPerWcnn,
                                                                WPerWcnn,
                                                                FilterSize,
                                                                DilationX,
                                                                DilationY,
                                                                Iters,
                                                                Aco,
                                                                UseF32I32>{};
    static constexpr auto conv_instr         = wcnn_conv_selector.selected_conv;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    static __device__ void UnshuffleStridedConv2Data(const InDataVec inputQuad[4],
                                                     InDataVec unshuffleQuad[4])
    {
#if defined(__gfx13__)
        uint32_t tmp0, tmp1, tmp2, tmp3;
        const uint32_t* pIn = reinterpret_cast<const uint32_t*>(inputQuad);
        uint32_t* pOut      = reinterpret_cast<uint32_t*>(unshuffleQuad);

        if constexpr(HPerWcnn == 4 && WPerWcnn == 4)
        {
            static_assert(sizeof(InDataVec) == 4, "");
            // Unshuffle via permutes - stage 1: horizontal, pat_size=1 pat_num=3
            tmp0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, pIn[0], pIn[1], 0xb);
            tmp2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, pIn[2], pIn[3], 0xb);
            // Unshuffle via permutes - stage 2: vertical, pat_size=3 pat_num=1
            pOut[0] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[2], tmp0, tmp2, 0x19);
            pOut[1] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[3], tmp1, tmp3, 0x19);
        }
        else if constexpr(HPerWcnn == 4 && WPerWcnn == 2)
        {
            static_assert(sizeof(InDataVec) == 4, "");
            // Unshuffle via permutes - stage 1: horizontal, pat_size=1 pat_num=1
            tmp0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, pIn[0], pIn[1], 0x9);
            tmp2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, pIn[2], pIn[3], 0x9);
            // Unshuffle via permutes - stage 2: vertical, pat_size=3 pat_num=1
            pOut[0] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[1], tmp0, tmp2, 0x19);
            pOut[2] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[3], tmp1, tmp3, 0x19);
        }
        else
        {
            static_assert(sizeof(InDataVec) == 8, "");
            // unshuffle for sub-tile 0
            tmp0    = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, pIn[0], pIn[2], 0xb);
            tmp2    = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, pIn[1], pIn[3], 0xb);
            pOut[0] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[4], tmp0, tmp2, 0x19);
            pOut[2] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[6], tmp1, tmp3, 0x19);

            // unshuffle for sub-tile 1
            tmp0    = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, pIn[4], pIn[6], 0xb);
            tmp2    = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, pIn[5], pIn[7], 0xb);
            pOut[1] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[5], tmp0, tmp2, 0x19);
            pOut[3] = __builtin_amdgcn_permute_pair_2src_interleave_b64(&pOut[7], tmp1, tmp3, 0x19);
        }
#else
        ignore = inputQuad;
        ignore = unshuffleQuad;
#endif
    }

    template <typename OutDataVec_>
    static __device__ void ShuffleConv2TransposedData(OutDataVec_* quadPtr[4])
    {
#if defined(__gfx13__)
        uint32_t tmp0, tmp1, tmp2, tmp3;
        static_assert(sizeof(OutDataVec_) % sizeof(uint32_t) == 0, "");
        constexpr index_t OutDataDw = sizeof(OutDataVec_) / sizeof(uint32_t);
        constexpr index_t NumLoop   = (HPerWcnn == 8 && WPerWcnn == 4) ? OutDataDw / 2 : OutDataDw;

        static_for<0, NumLoop, 1>{}([&](auto i) {
            uint32_t* p0 = reinterpret_cast<uint32_t*>(quadPtr[0]) + i;
            uint32_t* p1 = reinterpret_cast<uint32_t*>(quadPtr[1]) + i;
            uint32_t* p2 = reinterpret_cast<uint32_t*>(quadPtr[2]) + i;
            uint32_t* p3 = reinterpret_cast<uint32_t*>(quadPtr[3]) + i;
            if constexpr(HPerWcnn == 4 && WPerWcnn == 4)
            {
                // shuffle via permutes - stage 1: horizontal, pat_size=0 pat_num=0
                tmp0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, *p0, *p1, 0);
                tmp2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, *p2, *p3, 0);
                // shuffle via permutes - stage 2: vertical, pat_size=2 pat_num=1
                *p0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p1, tmp0, tmp2, 0x11);
                *p2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p3, tmp1, tmp3, 0x11);
            }
            else if constexpr(HPerWcnn == 4 && WPerWcnn == 2)
            {
                // shuffle via permutes - stage 1: horizontal, pat_size=1 pat_num=0
                tmp0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, *p0, *p1, 8);
                tmp2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, *p2, *p3, 8);
                // shuffle via permutes - stage 2: vertical, pat_size=2 pat_num=1
                *p0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p1, tmp0, tmp2, 0x11);
                *p2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p3, tmp1, tmp3, 0x11);
            }

            else if constexpr(HPerWcnn == 8 && WPerWcnn == 4)
            {
                uint32_t* p0_1 = p0 + NumLoop;
                uint32_t* p1_1 = p1 + NumLoop;
                uint32_t* p2_1 = p2 + NumLoop;
                uint32_t* p3_1 = p3 + NumLoop;
                uint32_t tmp4, tmp5, tmp6, tmp7;
                // shuffle via permutes - stage 1: horizontal, pat_size=0 pat_num=0
                tmp0 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp1, *p0, *p1, 0);
                tmp2 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp3, *p2, *p3, 0);
                tmp4 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp5, *p0_1, *p1_1, 0);
                tmp6 = __builtin_amdgcn_permute_pair_2src_interleave_b64(&tmp7, *p2_1, *p3_1, 0);

                // shuffle via permutes - stage 2: vertical, pat_size=2 pat_num=1
                *p0   = __builtin_amdgcn_permute_pair_2src_interleave_b64(p1, tmp0, tmp2, 0x11);
                *p0_1 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p1_1, tmp1, tmp3, 0x11);
                *p2   = __builtin_amdgcn_permute_pair_2src_interleave_b64(p3, tmp4, tmp6, 0x11);
                *p2_1 = __builtin_amdgcn_permute_pair_2src_interleave_b64(p3_1, tmp5, tmp7, 0x11);
            }

            else
            {
                static_assert(false, "never called");
            }
        });
#else
        ignore = quadPtr;
#endif
    }
#pragma clang diagnostic pop
};

} // namespace ck

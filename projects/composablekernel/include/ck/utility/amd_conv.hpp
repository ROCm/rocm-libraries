// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_CONV_HPP
#define CK_AMD_CONV_HPP

#include "data_type.hpp"
namespace ck {

#if defined(__gfx13__)
// Helper functions to translate convolution parameters to MOD flags
// __builtin_amdgcn_convolve* merged 3 MOD flags into single aux_data
// aux_data bit 0  ~ 5:  MOD0
// aux_data bit 6  ~ 17: MOD1    VOP5M Filter 1x1
//          bit 6  ~ 13: MOD1    VOP6M Filter 3x3
// aux_data bit 18 ~ 21: MOD2    VOP5M Filter 1x1
//          bit 18 ~ 25: MOD2    VOP6M Filter 3x3

constexpr index_t ConvolutionAuxData_Mod0 = 0;
constexpr index_t ConvolutionAuxData_Mod1 = 6;
constexpr index_t ConvolutionAuxData_Mod2 = 18;

// Filters
template <index_t FilterSize>
constexpr int32_t GetFilterSizeMod()
{
    // MOD0[5:3]
    return (FilterSize == 3) ? (1 << (3 + ConvolutionAuxData_Mod0)) : 0;
}

template <bool StartLane16>
constexpr int32_t GetStartLaneMod()
{
    // MOD2[2]
    return StartLane16 ? (1 << (2 + ConvolutionAuxData_Mod1)) : 0;
}

template <bool Signed>
constexpr int32_t GetWeightSignedMod()
{
    // MOD2[0]
    return Signed ? (1 << ConvolutionAuxData_Mod2) : 0;
}

template <index_t DilationX, index_t DilationY>
constexpr int32_t GetDilationMod()
{
    // MOD1[1:0]
    return ((DilationX == 2) ? (1 << ConvolutionAuxData_Mod1) : 0) |
           ((DilationY == 2) ? (2 << ConvolutionAuxData_Mod1) : 0);
}

// Accumulator
template <bool IsBias>
constexpr int32_t GetAccumIsBiasMod()
{
    // MOD2[1]
    return IsBias ? (1 << (1 + ConvolutionAuxData_Mod2)) : 0;
}

template <bool Aco>
constexpr int32_t GetAccumChannelOrderMod()
{
    // MOD1[3]
    return Aco ? (1 << (3 + ConvolutionAuxData_Mod1)) : 0;
}

template <bool IntScale>
constexpr int32_t GeIntScaleMod()
{
    // MOD2[3]
    return IntScale ? (1 << (3 + ConvolutionAuxData_Mod2)) : 0;
}

// Tensor
template <bool Signed>
constexpr int32_t GetTensorSignedMod()
{
    // MOD2[2]
    return Signed ? (1 << (2 + ConvolutionAuxData_Mod2)) : 0;
}

template <index_t Iters>
constexpr int32_t GetItersMod()
{
    // MOD1[7:6], 1x1 only
    static_assert(Iters <= 4, "Filter 1x1 only support Iters 0~3!!");
    return Iters ? ((Iters - 1) << (6 + ConvolutionAuxData_Mod1)) : 0;
}

// FilterSize = 1 x 1
// src: fp16, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_f16;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f16<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_wei, const half2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_f16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f16<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_wei, const half2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_f16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f16<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_wei, const half2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_f16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: bf16, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_bf16;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf16<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_wei, const bhalf2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf16<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_wei, const bhalf2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf16<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_wei, const bhalf2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: f8, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_f8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: bf8, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_bf8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: f8, bf8, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_f8_bf8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8_bf8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8_bf8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8_bf8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_bf8_f8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8_f8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8_f8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8_f8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: iu8, dst: fp32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_iu8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_wei, const int8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_wei, const int8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32i32_iu8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu8_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_wei, const int8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu8_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_wei, const int8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu8_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

// src: iu8, dst: i32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_i32_iu8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu8_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_wei, const int8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu8_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_wei, const int8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu8_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
// src: iu4, dst: f32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32_iu4;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu4<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x16_t& reg_wei, const int4x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu4_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu4<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x32_t& reg_wei, const int4x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu4_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu4<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x64_t& reg_wei, const int4x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu4_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f32i32_iu4;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu4<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x16_t& reg_wei, const int4x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu4_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu4<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x32_t& reg_wei, const int4x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu4_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu4<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x64_t& reg_wei, const int4x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu4_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};
// src: iu4, dst: i32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_i32_iu4;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu4<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x16_t& reg_wei, const int4x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu4_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu4<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x32_t& reg_wei, const int4x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu4_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu4<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x64_t& reg_wei, const int4x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu4_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};
#endif

// src: f16, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_f16;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_wei, const half2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_wei, const half2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_wei, const half2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half2_t& reg_wei, const half2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_wei, const half2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_wei, const half2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half2_t& reg_wei, const half4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half2_t& reg_wei, const half4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_wei, const half4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: bf16, dst: bf16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_bf16_bf16;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_wei, const bhalf2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x2(
            reg_c.template AsType<bhalf4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_wei, const bhalf2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x2(
            reg_c.template AsType<bhalf4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_wei, const bhalf2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x2(
            reg_c.template AsType<bhalf4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_wei, const bhalf2_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_wei, const bhalf2_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_wei, const bhalf2_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_wei, const bhalf4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_8x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_wei, const bhalf4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_8x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_wei, const bhalf4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_8x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[0],
            *reg_data[1],
            *reg_data[2],
            *reg_data[3],
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: bf8, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_bf8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) =
            bit_cast<half4_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x2(
                bit_cast<half4_t>(reg_c.template AsType<half4_t>()[Number<0>{}]),
                bit_cast<int32x2_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) =
            bit_cast<half4_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x2(
                bit_cast<half4_t>(reg_c.template AsType<half4_t>()[Number<0>{}]),
                bit_cast<int32x4_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                bit_cast<int32_t>(*reg_data[1]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<2>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) =
            bit_cast<half4_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x2(
                bit_cast<half4_t>(reg_c.template AsType<half4_t>()[Number<0>{}]),
                bit_cast<int32x8_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                bit_cast<int32_t>(*reg_data[1]),
                bit_cast<int32_t>(*reg_data[2]),
                bit_cast<int32_t>(*reg_data[3]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<4>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32x2_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                bit_cast<int32_t>(*reg_data[1]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<2>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32x4_t>(reg_wei),
                bit_cast<int32_t>(*reg_data[0]),
                bit_cast<int32_t>(*reg_data[1]),
                bit_cast<int32_t>(*reg_data[2]),
                bit_cast<int32_t>(*reg_data[3]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<4>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const bf8x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_8x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32_t>(reg_wei),
                bit_cast<int32x2_t>(*reg_data[0]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const bf8x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_8x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32_t>(reg_wei),
                bit_cast<int32x2_t>(*reg_data[0]),
                bit_cast<int32x2_t>(*reg_data[1]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<2>(),
                0));
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const bf8x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_8x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32x2_t>(reg_wei),
                bit_cast<int32x2_t>(*reg_data[0]),
                bit_cast<int32x2_t>(*reg_data[1]),
                bit_cast<int32x2_t>(*reg_data[2]),
                bit_cast<int32x2_t>(*reg_data[3]),
                GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() |
                    GetStartLaneMod<HighLane>() | GetItersMod<4>(),
                0));
    }
};

// src: f8, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_f8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const f8x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const f8x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const f8x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            bit_cast<int32x2_t>(*reg_data[2]),
            bit_cast<int32x2_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: f8, bf8, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_f8_bf8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const bf8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const bf8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_wei, const bf8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const bf8x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x4_t& reg_wei, const bf8x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_wei, const bf8x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            bit_cast<int32x2_t>(*reg_data[2]),
            bit_cast<int32x2_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: bf8, f8, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_bf8_f8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const f8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const f8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_wei, const f8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const f8x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x4_t& reg_wei, const f8x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_wei, const f8x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            bit_cast<int32x2_t>(*reg_data[2]),
            bit_cast<int32x2_t>(*reg_data[3]),
            GetFilterSizeMod<1>() | GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() |
                GetItersMod<4>(),
            0);
    }
};

// src: iu8, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_iu8;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_wei, const int8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_wei, const int8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_wei, const int8x4_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x4_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_wei, const int8x4_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_wei, const int8x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_wei, const int8x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_wei, const int8x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            bit_cast<int32x2_t>(*reg_data[2]),
            bit_cast<int32x2_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
// src: iu4, dst: f16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false,
          bool HighLane = false>
struct intrin_wcnn_conv_f16_iu4;

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 2, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x16_t& reg_wei, const int4x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 2, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x32_t& reg_wei, const int4x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 2, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x64_t& reg_wei, const int4x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x8_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x8_t& reg_wei, const int4x8_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x16_t& reg_wei, const int4x8_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x32_t& reg_wei, const int4x8_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x4_t>(reg_wei),
            bit_cast<int32_t>(*reg_data[0]),
            bit_cast<int32_t>(*reg_data[1]),
            bit_cast<int32_t>(*reg_data[2]),
            bit_cast<int32_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<8, 4, 1, 1, 1, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x8_t& reg_wei, const int4x16_t* reg_data[1], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<8, 4, 1, 1, 1, 2, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const int4x8_t& reg_wei, const int4x16_t* reg_data[2], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<2>(),
            0);
    }
};

template <bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<8, 4, 1, 1, 1, 4, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x16_t& reg_wei, const int4x16_t* reg_data[4], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x2_t>(reg_wei),
            bit_cast<int32x2_t>(*reg_data[0]),
            bit_cast<int32x2_t>(*reg_data[1]),
            bit_cast<int32x2_t>(*reg_data[2]),
            bit_cast<int32x2_t>(*reg_data[3]),
            GetTensorSignedMod<Signed>() | GetWeightSignedMod<Signed>() | GetFilterSizeMod<1>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>() | GetItersMod<4>(),
            0);
    }
};
#endif

// FilterSize = 3x3
// src: fp16, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f16<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half36_t& reg_wei, const half6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_f16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *(reg_data[1]),
            *(reg_data[0]),
            *(reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf16, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf16<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf36_t& reg_wei, const bhalf6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf16_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: f8, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x72_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: f8, bf8, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_f8_bf8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x72_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_fp8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf8, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x72_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_bf8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf8, fp8, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_bf8_f8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x72_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_bf8_fp8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu8, dst: fp32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x72_t& reg_wei, const int8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu8_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu8, dst: fp32i32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x72_t& reg_wei, const int8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu8_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu8, dst: i32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x72_t& reg_wei, const int8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu8_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
// src: iu4, dst: f32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32_iu4<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x144_t& reg_wei, const int4x24_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32_iu4_4x2(
            reg_c.template AsType<float4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu4, dst: f32i32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f32i32_iu4<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x144_t& reg_wei, const int4x24_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f32i32_iu4_4x2(
            bit_cast<int32x4_t>(reg_c.template AsType<float4_t>()[Number<0>{}]),
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu4, dst: i32
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_i32_iu4<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x144_t& reg_wei, const int4x24_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_i32_iu4_4x2(
            reg_c.template AsType<int32x4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};
#endif

// src: f16, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half36_t& reg_wei, const half6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half18_t& reg_wei, const half6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f16<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const half10_t& reg_wei, const half8_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_f16_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf16, dst: bf16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf36_t& reg_wei, const bhalf6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x2(
            reg_c.template AsType<bhalf4_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf18_t& reg_wei, const bhalf6_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_4x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_bf16_bf16<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bhalf10_t& reg_wei, const bhalf8_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_bf16_bf16_8x4(
            reg_c.template AsType<bhalf8_t>()[Number<0>{}],
            reg_wei,
            *reg_data[1],
            *reg_data[0],
            *reg_data[2],
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf8, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x72_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) =
            bit_cast<half4_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x2(
                bit_cast<half4_t>(reg_c.template AsType<half4_t>()[Number<0>{}]),
                bit_cast<int32x18_t>(reg_wei),
                bit_cast<int32x3_t>(*reg_data[1]),
                bit_cast<int32x3_t>(*reg_data[0]),
                bit_cast<int32x3_t>(*reg_data[2]),
                GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                    GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
                0));
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x36_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_4x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32x9_t>(reg_wei),
                bit_cast<int32x3_t>(*reg_data[1]),
                bit_cast<int32x3_t>(*reg_data[0]),
                bit_cast<int32x3_t>(*reg_data[2]),
                GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                    GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
                0));
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x20_t& reg_wei, const bf8x16_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            bit_cast<half8_t>(__builtin_amdgcn_convolve_f16_bf8_bf8_8x4(
                bit_cast<half8_t>(reg_c.template AsType<half8_t>()[Number<0>{}]),
                bit_cast<int32x5_t>(reg_wei),
                bit_cast<int32x4_t>(*reg_data[1]),
                bit_cast<int32x4_t>(*reg_data[0]),
                bit_cast<int32x4_t>(*reg_data[2]),
                GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                    GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
                0));
    }
};

// src: f8, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x72_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x36_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x9_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x20_t& reg_wei, const f8x16_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x5_t>(reg_wei),
            bit_cast<int32x4_t>(*reg_data[1]),
            bit_cast<int32x4_t>(*reg_data[0]),
            bit_cast<int32x4_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: f8, bf8, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x72_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x36_t& reg_wei, const bf8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x9_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_f8_bf8<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const f8x20_t& reg_wei, const bf8x16_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_fp8_bf8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x5_t>(reg_wei),
            bit_cast<int32x4_t>(*reg_data[1]),
            bit_cast<int32x4_t>(*reg_data[0]),
            bit_cast<int32x4_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: bf8, f8, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x72_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x36_t& reg_wei, const f8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x9_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_bf8_f8<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void Run(const bf8x20_t& reg_wei, const f8x16_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_bf8_fp8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x5_t>(reg_wei),
            bit_cast<int32x4_t>(*reg_data[1]),
            bit_cast<int32x4_t>(*reg_data[0]),
            bit_cast<int32x4_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

// src: iu8, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x72_t& reg_wei, const int8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x36_t& reg_wei, const int8x12_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x9_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu8<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int8x20_t& reg_wei, const int8x16_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu8_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x5_t>(reg_wei),
            bit_cast<int32x4_t>(*reg_data[1]),
            bit_cast<int32x4_t>(*reg_data[0]),
            bit_cast<int32x4_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
// src: iu4, dst: f16
template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 2, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x144_t& reg_wei, const int4x24_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half4_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x2(
            reg_c.template AsType<half4_t>()[Number<0>{}],
            bit_cast<int32x18_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<4, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x72_t& reg_wei, const int4x24_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_4x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x9_t>(reg_wei),
            bit_cast<int32x3_t>(*reg_data[1]),
            bit_cast<int32x3_t>(*reg_data[0]),
            bit_cast<int32x3_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};

template <index_t DilationX, index_t DilationY, bool Aco, bool Signed, bool HighLane>
struct intrin_wcnn_conv_f16_iu4<8, 4, 3, DilationX, DilationY, 1, Aco, Signed, HighLane>
{
    template <class FloatC>
    __device__ static void
    Run(const int4x40_t& reg_wei, const int4x32_t* reg_data[3], FloatC& reg_c)
    {
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_convolve_f16_iu4_8x4(
            reg_c.template AsType<half8_t>()[Number<0>{}],
            bit_cast<int32x5_t>(reg_wei),
            bit_cast<int32x4_t>(*reg_data[1]),
            bit_cast<int32x4_t>(*reg_data[0]),
            bit_cast<int32x4_t>(*reg_data[2]),
            GetDilationMod<DilationX, DilationY>() | GetTensorSignedMod<Signed>() |
                GetWeightSignedMod<Signed>() | GetFilterSizeMod<3>() |
                GetAccumChannelOrderMod<Aco>() | GetStartLaneMod<HighLane>(),
            0);
    }
};
#endif

#endif

} // namespace ck
#endif

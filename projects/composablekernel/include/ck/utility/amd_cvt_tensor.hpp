// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_CVT_TENSOR_HPP
#define CK_AMD_CVT_TENSOR_HPP

#include "data_type.hpp"
namespace ck {

// __builtin_amdgcn_scale_bias_activate_f32
// aux_data[ 5: 0] maps to instr[ 63: 58] for VOP2M-VOP6M (MOD0)
// aux_data[17: 6] maps to instr[ 87: 76] for VOP3M-VOP5M (MOD1)
// aux_data[13: 6] maps to instr[ 83: 76] for VOP6M       (MOD1)
// aux_data[25:18] maps to instr[127:120] for VOP5M-VOP6M (MOD2)

// MOD0[2:0] PIXEL_SHAPE[3]
//  0 : 8x4 with 8 OC, 32 pixels in 4x 16b accumulator VGPRs
//  1 : 4x4 with 16 OC, 16 pixels in 4x 16b accumulator VGPRs
//  2 : 4x4 with 8 OC,16 pixels in 2x 16b accumulator VGPRs.(V_CVT_TO_TENSOR - only shape)
//  3 : 4x2 with 16 OC,8 pixels in 2x 16b accumulator VGPRs, or 4 - 32b accumulator VGPRs when 32bit
//  accumulation is used 4 -7: reserved.

// MOD1[1] Activation function: applied after scale and bias
// 0 None result = input
// 1 ReLU result = MAX (0, input)

// auxData = (activateFun & 0x180) | (pixelShape & 0x7) ;
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i4_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i4_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_f16_4x4x16(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i4_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_bf16_4x4x16(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i4_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u4_f32;
// u4 input tensor
template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u4_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_f16_4x4x16(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u4_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_bf16_4x4x16(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u4_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u4_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

// i8 input tensor
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i8_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i8_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i8_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i8_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_i8_f16_4x4x16(
            out_0, out_1, inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i8_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_i8_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i8_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_i8_bf16_4x4x16(
            out_0, out_1, inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_i8_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u8_f32;
// u8 input tensor
template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u8_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u8_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u8_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_u8_f16_4x4x16(
            out_0, out_1, inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u8_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_u8_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u8_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_u8_bf16_4x4x16(
            out_0, out_1, inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_u8_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

// fp8 input tensor
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp8_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_fp8_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp8_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_fp8_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_fp8_f16_4x4x16(
            out_0, out_1, inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_fp8_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp8_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_fp8_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_fp8_bf16_4x4x16(
            out_0, out_1, inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_fp8_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

// bf8 input tensor
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf8_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_bf8_f32_4x2x16(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf8_fp16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_bf8_f16_8x4x8(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf8_f16_4x4x16(
            out_0, out_1, inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_bf8_f16_4x4x8_4x2x16(
            inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf8_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int32x2_t& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_bf8_bf16_8x4x8(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int* out_0, int* out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf8_bf16_4x4x16(
            out_0, out_1, inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const char& ssrc, int& out_0)
    {
        out_0 = __builtin_amdgcn_cvt_to_tensor_bf8_bf16_4x4x8_4x2x16(
            inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

// fp16 input tensor
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp16_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, half2_t& out_0, half2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_f32_4x2x16(
            reinterpret_cast<fp16x2_t*>(&out_0),
            reinterpret_cast<fp16x2_t*>(&out_1),
            inAcc.template AsType<float4_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp16_fp16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, half4_t& out_0, half4_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_f16_8x4x8(reinterpret_cast<fp16x4_t*>(&out_0),
                                                     reinterpret_cast<fp16x4_t*>(&out_1),
                                                     inAcc.template AsType<half8_t>()[Number<0>{}],
                                                     ssrc,
                                                     AuxData,
                                                     Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const char& ssrc,
                               half2_t& out_0,
                               half2_t& out_1,
                               half2_t& out_2,
                               half2_t& out_3)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_f16_4x4x16(reinterpret_cast<fp16x2_t*>(&out_0),
                                                      reinterpret_cast<fp16x2_t*>(&out_1),
                                                      reinterpret_cast<fp16x2_t*>(&out_2),
                                                      reinterpret_cast<fp16x2_t*>(&out_3),
                                                      inAcc.template AsType<half8_t>()[Number<0>{}],
                                                      ssrc,
                                                      AuxData,
                                                      Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, half2_t& out_0, half2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_f16_4x4x8_4x2x16(
            reinterpret_cast<fp16x2_t*>(&out_0),
            reinterpret_cast<fp16x2_t*>(&out_1),
            inAcc.template AsType<half4_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_fp16_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, half4_t& out_0, half4_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_bf16_8x4x8(
            reinterpret_cast<fp16x4_t*>(&out_0),
            reinterpret_cast<fp16x4_t*>(&out_1),
            inAcc.template AsType<bhalf8_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const char& ssrc,
                               half2_t& out_0,
                               half2_t& out_1,
                               half2_t& out_2,
                               half2_t& out_3)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_bf16_4x4x16(
            reinterpret_cast<fp16x2_t*>(&out_0),
            reinterpret_cast<fp16x2_t*>(&out_1),
            reinterpret_cast<fp16x2_t*>(&out_2),
            reinterpret_cast<fp16x2_t*>(&out_3),
            inAcc.template AsType<bhalf8_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, half2_t& out_0, half2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_f16_bf16_4x4x8_4x2x16(
            reinterpret_cast<fp16x2_t*>(&out_0),
            reinterpret_cast<fp16x2_t*>(&out_1),
            inAcc.template AsType<bhalf4_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

// bf16 input tensor
template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf16_f32;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_f32<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, bhalf2_t& out_0, bhalf2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_f32_4x2x16(
            &out_0, &out_1, inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf16_f16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_f16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, bhalf4_t& out_0, bhalf4_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_f16_8x4x8(
            &out_0, &out_1, inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_f16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const char& ssrc,
                               bhalf2_t& out_0,
                               bhalf2_t& out_1,
                               bhalf2_t& out_2,
                               bhalf2_t& out_3)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_f16_4x4x16(
            &out_0,
            &out_1,
            &out_2,
            &out_3,
            inAcc.template AsType<half8_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_f16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, bhalf2_t& out_0, bhalf2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_f16_4x4x8_4x2x16(
            &out_0, &out_1, inAcc.template AsType<half4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp, index_t H, index_t W>
struct intrin_wcnn_cvt_tensor_bf16_bf16;

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 8, 4>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, bhalf4_t& out_0, bhalf4_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_bf16_8x4x8(
            &out_0, &out_1, inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 4, 4>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const char& ssrc,
                               bhalf2_t& out_0,
                               bhalf2_t& out_1,
                               bhalf2_t& out_2,
                               bhalf2_t& out_3)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_bf16_4x4x16(
            &out_0,
            &out_1,
            &out_2,
            &out_3,
            inAcc.template AsType<bhalf8_t>()[Number<0>{}],
            ssrc,
            AuxData,
            Clamp);
    }
};

template <index_t AuxData, bool Clamp>
struct intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 4, 2>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const char& ssrc, bhalf2_t& out_0, bhalf2_t& out_1)
    {
        __builtin_amdgcn_cvt_to_tensor_bf16_bf16_4x4x8_4x2x16(
            &out_0, &out_1, inAcc.template AsType<bhalf4_t>()[Number<0>{}], ssrc, AuxData, Clamp);
    }
};

} // namespace ck
#endif

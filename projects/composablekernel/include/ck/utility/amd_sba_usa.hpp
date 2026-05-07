// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_SBA_HPP
#define CK_AMD_SBA_HPP

#include "dtype_vector.hpp"
namespace ck {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"

// __builtin_amdgcn_scale_bias_activate_f32
// aux_data[ 5: 0] maps to instr[ 63: 58] for VOP2M-VOP6M (MOD0)
// aux_data[17: 6] maps to instr[ 87: 76] for VOP3M-VOP5M (MOD1)
// aux_data[13: 6] maps to instr[ 83: 76] for VOP6M       (MOD1)
// aux_data[25:18] maps to instr[127:120] for VOP5M-VOP6M (MOD2)

// MOD1[3:2] Activation function: applied after scale and bias
// 0 None result = input
// 1 ReLU result = MAX (0, input)
// 2 HardTanH result = MIN(1, MAX(-1, input)) // clamp input to [-1.. 1]
// 3 reserved undefined

// the activation function is encoded in bits AuxData[9:8] in the clang builtin.
// For example, if you want to use ReLU which is 1, you need to set AuxData[8] == 256.

// auxData = (activateFun & 0x180) | (pixelShape & 0x7) ;
template <index_t auxdata, bool scalebiasPacked, bool uniformScale>
struct intrin_wcnn_sba_f32;

template <index_t auxdata>
struct intrin_wcnn_sba_f32<auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const float& ssrc, const float& bias, FloatAcc& outAcc)
    {
        outAcc.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_f32(
            inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, bias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_f32<auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const float& scalebias, FloatAcc& outAcc)
    {
        outAcc.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_f32(
            inAcc.template AsType<float4_t>()[Number<0>{}], 0, scalebias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_f32<auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const float& ssrc, FloatAcc& outAcc)
    {
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_uniform_scale_activate_f32(
                inAcc.template AsType<float4_t>()[Number<0>{}], ssrc, auxdata, true);
    }
};

template <index_t auxdata, bool scalebiasPacked, bool uniformScale>
struct intrin_wcnn_sba_half;

template <index_t auxdata>
struct intrin_wcnn_sba_half<auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const half_t& ssrc, const half2_t& bias, FloatAcc& outAcc)
    {
        outAcc.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_f16(
            inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, bias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_half<auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const half2_t& scaleBias, FloatAcc& outAcc)
    {
        outAcc.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_f16(
            inAcc.template AsType<half8_t>()[Number<0>{}], 0, scaleBias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_half<auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const half_t& ssrc, FloatAcc& outAcc)
    {
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_uniform_scale_activate_f16(
                inAcc.template AsType<half8_t>()[Number<0>{}], ssrc, auxdata, true);
    }
};

template <index_t auxdata, bool scalebiasPacked, bool uniformScale>
struct intrin_wcnn_sba_scatter2_half;

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_half<auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const half_t& ssrc,
                               const half2_t& bias,
                               half2_t& outAcc0,
                               half2_t& outAcc1)
    {
        __builtin_amdgcn_scale_bias_activate_scatter2_f16(
            reinterpret_cast<fp16x2_t*>(&outAcc0),
            reinterpret_cast<fp16x2_t*>(&outAcc1),
            inAcc.template AsType<half4_t>()[Number<0>{}],
            ssrc,
            bias,
            auxdata,
            true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_half<auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const half2_t& scale_bias, half2_t& outAcc0, half2_t& outAcc1)
    {
        __builtin_amdgcn_scale_bias_activate_scatter2_f16(
            reinterpret_cast<fp16x2_t*>(&outAcc0),
            reinterpret_cast<fp16x2_t*>(&outAcc1),
            inAcc.template AsType<half4_t>()[Number<0>{}],
            0,
            scale_bias,
            auxdata,
            true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_half<auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const half_t& ssrc, half2_t& outAcc0, half2_t& outAcc1)
    {
        __builtin_amdgcn_uniform_scale_activate_scatter2_f16(
            reinterpret_cast<fp16x2_t*>(&outAcc0),
            reinterpret_cast<fp16x2_t*>(&outAcc1),
            inAcc.template AsType<half4_t>()[Number<0>{}],
            ssrc,
            auxdata,
            true);
    }
};

template <index_t auxdata, bool scalebiasPacked, bool uniformScale>
struct intrin_wcnn_sba_bhalf;

template <index_t auxdata>
struct intrin_wcnn_sba_bhalf<auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const bhalf_t& ssrc, const bhalf2_t& bias, FloatAcc& outAcc)
    {
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_bf16(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, bias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_bhalf<auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const bhalf2_t& scaleBias, FloatAcc& outAcc)
    {
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_scale_bias_activate_bf16(
            inAcc.template AsType<bhalf8_t>()[Number<0>{}], 0, scaleBias, auxdata, true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_bhalf<auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc, const bhalf_t& ssrc, FloatAcc& outAcc)
    {
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_uniform_scale_activate_bf16(
                inAcc.template AsType<bhalf8_t>()[Number<0>{}], ssrc, auxdata, true);
    }
};

template <index_t auxdata, bool scalebiasPacked, bool uniformScale>
struct intrin_wcnn_sba_scatter2_bhalf;

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_bhalf<auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ static void Run(const FloatAcc& inAcc,
                               const bhalf_t& ssrc,
                               const bhalf2_t& bias,
                               bhalf2_t& outAcc0,
                               bhalf2_t& outAcc1)
    {
        __builtin_amdgcn_scale_bias_activate_scatter2_bf16(
            &outAcc0,
            &outAcc1,
            inAcc.template AsType<bhalf4_t>()[Number<0>{}],
            ssrc,
            bias,
            auxdata,
            true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_bhalf<auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const bhalf2_t& scale_bias, bhalf2_t& outAcc0, bhalf2_t& outAcc1)
    {
        __builtin_amdgcn_scale_bias_activate_scatter2_bf16(
            &outAcc0,
            &outAcc1,
            inAcc.template AsType<bhalf4_t>()[Number<0>{}],
            0,
            scale_bias,
            auxdata,
            true);
    }
};

template <index_t auxdata>
struct intrin_wcnn_sba_scatter2_bhalf<auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ static void
    Run(const FloatAcc& inAcc, const bhalf_t& ssrc, bhalf2_t& outAcc0, bhalf2_t& outAcc1)
    {
        __builtin_amdgcn_uniform_scale_activate_scatter2_bf16(
            &outAcc0,
            &outAcc1,
            inAcc.template AsType<bhalf4_t>()[Number<0>{}],
            ssrc,
            auxdata,
            true);
    }
};

#pragma clang diagnostic pop

} // namespace ck
#endif

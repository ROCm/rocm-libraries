// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_cvt_tensor.hpp"

namespace ck {

enum struct WcnnCvtTensorInstr
{
    // gfx13
    wcnn_cvt_tensor_i4_f32 = 0,
    wcnn_cvt_tensor_i4_fp16,
    wcnn_cvt_tensor_i4_bf16,
    wcnn_cvt_tensor_u4_f32,
    wcnn_cvt_tensor_u4_fp16,
    wcnn_cvt_tensor_u4_bf16,
    wcnn_cvt_tensor_i8_f32,
    wcnn_cvt_tensor_i8_fp16,
    wcnn_cvt_tensor_i8_bf16,
    wcnn_cvt_tensor_u8_f32,
    wcnn_cvt_tensor_u8_fp16,
    wcnn_cvt_tensor_u8_bf16,
    wcnn_cvt_tensor_fp8_f32,
    wcnn_cvt_tensor_fp8_fp16,
    wcnn_cvt_tensor_fp8_bf16,
    wcnn_cvt_tensor_bf8_f32,
    wcnn_cvt_tensor_bf8_half,
    wcnn_cvt_tensor_bf8_bhalf,
    wcnn_cvt_tensor_fp16_f32,
    wcnn_cvt_tensor_fp16_fp16,
    wcnn_cvt_tensor_fp16_bf16,
    wcnn_cvt_tensor_bf16_f32,
    wcnn_cvt_tensor_bf16_fp16,
    wcnn_cvt_tensor_bf16_bf16
};

template <WcnnCvtTensorInstr Instr, index_t AuxData, bool Clamp, index_t H, index_t W>
struct WcnnCvtTensorType
{
    WcnnCvtTensorType() { static_assert(false, "never called"); }
};

// i4_f32
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i4_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// i4_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0); // int32x2->int4x16
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 4, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i4_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// i4_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0); // int32x2->int4x16
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 4, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// u4_f32
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u4_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// u4_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 4, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u4_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// u4_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_i4_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0); // int32x2->int4x16
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u4_bf16<AuxData, Clamp, 4, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u4_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i8_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// i8_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0); // int32x2_t->int8x8t
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        int dwordOut_0 = 0;
        int dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<int8x4_t>()(Number<0>{}) = bit_cast<int8x4_t>(dwordOut_0);
        out.template AsType<int8x4_t>()(Number<1>{}) = bit_cast<int8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i8_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// i8_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<int8x4_t>()(Number<0>{}) = bit_cast<int8x4_t>(dwordOut_0);
        out.template AsType<int8x4_t>()(Number<1>{}) = bit_cast<int8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_i8_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u8_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// u8_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0); // int32x2_t->int8x8t
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<uint8x4_t>()(Number<0>{}) = bit_cast<uint8x4_t>(dwordOut_0);
        out.template AsType<uint8x4_t>()(Number<1>{}) = bit_cast<uint8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u8_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// u8_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<uint8x4_t>()(Number<0>{}) = bit_cast<uint8x4_t>(dwordOut_0);
        out.template AsType<uint8x4_t>()(Number<1>{}) = bit_cast<uint8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_u8_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_fp8_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// fp8_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<f8x4_t>()(Number<0>{}) = bit_cast<f8x4_t>(dwordOut_0);
        out.template AsType<f8x4_t>()(Number<1>{}) = bit_cast<f8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_fp8_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// fp8_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<f8x4_t>()(Number<0>{}) = bit_cast<f8x4_t>(dwordOut_0);
        out.template AsType<f8x4_t>()(Number<1>{}) = bit_cast<f8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_fp8_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_bf8_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// bf8_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_half, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_half, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<bf8x4_t>()(Number<0>{}) = bit_cast<bf8x4_t>(dwordOut_0);
        out.template AsType<bf8x4_t>()(Number<1>{}) = bit_cast<bf8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_half, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_bf8_fp16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

// bf8_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_bhalf, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        int32x2_t dwordOut_0;
        intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_bhalf, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        auto dwordOut_1 = 0;
        intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, &dwordOut_0, &dwordOut_1);
        out.template AsType<bf8x4_t>()(Number<0>{}) = bit_cast<bf8x4_t>(dwordOut_0);
        out.template AsType<bf8x4_t>()(Number<1>{}) = bit_cast<bf8x4_t>(dwordOut_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_bhalf, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out_0) const
    {
#if defined(__gfx13__)
        auto dwordOut_0 = 0;
        intrin_wcnn_cvt_tensor_bf8_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, dwordOut_0);
        out_0 = bit_cast<OutTensor>(dwordOut_0);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out_0;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half2_t& out_0 = out.template AsType<half2_t>()(Number<0>{});
        half2_t& out_1 = out.template AsType<half2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

// fp16_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half4_t& out_0 = out.template AsType<half4_t>()(Number<0>{});
        half4_t& out_1 = out.template AsType<half4_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half2_t& out_0 = out.template AsType<half2_t>()(Number<0>{});
        half2_t& out_1 = out.template AsType<half2_t>()(Number<1>{});
        half2_t& out_2 = out.template AsType<half2_t>()(Number<2>{});
        half2_t& out_3 = out.template AsType<half2_t>()(Number<3>{});
        intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, out_0, out_1, out_2, out_3);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half2_t& out_0 = out.template AsType<half2_t>()(Number<0>{});
        half2_t& out_1 = out.template AsType<half2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_fp16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

// fp16_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half4_t& out_0 = out.template AsType<half4_t>()(Number<0>{});
        half4_t& out_1 = out.template AsType<half4_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half2_t& out_0 = out.template AsType<half2_t>()(Number<0>{});
        half2_t& out_1 = out.template AsType<half2_t>()(Number<1>{});
        half2_t& out_2 = out.template AsType<half2_t>()(Number<2>{});
        half2_t& out_3 = out.template AsType<half2_t>()(Number<3>{});
        intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, out_0, out_1, out_2, out_3);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        half2_t& out_0 = out.template AsType<half2_t>()(Number<0>{});
        half2_t& out_1 = out.template AsType<half2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_f32, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const float& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf2_t& out_0 = out.template AsType<bhalf2_t>()(Number<0>{});
        bhalf2_t& out_1 = out.template AsType<bhalf2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_bf16_f32<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

// bf16_fp16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_fp16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf4_t& out_0 = out.template AsType<bhalf4_t>()(Number<0>{});
        bhalf4_t& out_1 = out.template AsType<bhalf4_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_fp16_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_fp16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf2_t& out_0 = out.template AsType<bhalf2_t>()(Number<0>{});
        bhalf2_t& out_1 = out.template AsType<bhalf2_t>()(Number<1>{});
        bhalf2_t& out_2 = out.template AsType<bhalf2_t>()(Number<2>{});
        bhalf2_t& out_3 = out.template AsType<bhalf2_t>()(Number<3>{});
        intrin_wcnn_cvt_tensor_bf16_f16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, out_0, out_1, out_2, out_3);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_fp16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const half_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf2_t& out_0 = out.template AsType<bhalf2_t>()(Number<0>{});
        bhalf2_t& out_1 = out.template AsType<bhalf2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_bf16_f16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

// bf16_bf16
template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_bf16, AuxData, Clamp, 8, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf4_t& out_0 = out.template AsType<bhalf4_t>()(Number<0>{});
        bhalf4_t& out_1 = out.template AsType<bhalf4_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 8, 4>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_bf16, AuxData, Clamp, 4, 4>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf2_t& out_0 = out.template AsType<bhalf2_t>()(Number<0>{});
        bhalf2_t& out_1 = out.template AsType<bhalf2_t>()(Number<1>{});
        bhalf2_t& out_2 = out.template AsType<bhalf2_t>()(Number<2>{});
        bhalf2_t& out_3 = out.template AsType<bhalf2_t>()(Number<3>{});
        intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 4, 4>::Run(
            inAcc, scale, out_0, out_1, out_2, out_3);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <index_t AuxData, bool Clamp>
struct WcnnCvtTensorType<WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_bf16, AuxData, Clamp, 4, 2>
{
    template <class FloatAcc, class OutTensor>
    __device__ void Run(const FloatAcc& inAcc, const bhalf_t& scale, OutTensor& out) const
    {
#if defined(__gfx13__)
        bhalf2_t& out_0 = out.template AsType<bhalf2_t>()(Number<0>{});
        bhalf2_t& out_1 = out.template AsType<bhalf2_t>()(Number<1>{});
        intrin_wcnn_cvt_tensor_bf16_bf16<AuxData, Clamp, 4, 2>::Run(inAcc, scale, out_0, out_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = out;
#endif
    }
};

template <typename InDataType,
          typename AccDataType,
          index_t AuxData,
          bool Clamp,
          index_t HPerWcnn,
          index_t WPerWcnn>
struct WcnnCvtTensorSelector
{
    template <typename InDataType_, typename AccDataType_>
    static constexpr auto GetCvtTensor();

    template <>
    constexpr auto GetCvtTensor<int4_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_f32;
    }

    template <>
    constexpr auto GetCvtTensor<int4_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<int4_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i4_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<uint4_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_f32;
    }

    template <>
    constexpr auto GetCvtTensor<uint4_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<int4_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u4_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<int8_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_f32;
    }

    template <>
    constexpr auto GetCvtTensor<int8_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<int8_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_i8_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<uint8_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_f32;
    }

    template <>
    constexpr auto GetCvtTensor<uint8_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<uint8_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_u8_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<f8_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_f32;
    }

    template <>
    constexpr auto GetCvtTensor<f8_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<f8_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp8_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<bf8_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_f32;
    }

    template <>
    constexpr auto GetCvtTensor<bf8_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_half;
    }

    template <>
    constexpr auto GetCvtTensor<bf8_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf8_bhalf;
    }

    template <>
    constexpr auto GetCvtTensor<half_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_f32;
    }

    template <>
    constexpr auto GetCvtTensor<half_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<half_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_fp16_bf16;
    }

    template <>
    constexpr auto GetCvtTensor<bhalf_t, float_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_f32;
    }

    template <>
    constexpr auto GetCvtTensor<bhalf_t, half_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_fp16;
    }

    template <>
    constexpr auto GetCvtTensor<bhalf_t, bhalf_t>()
    {
        return WcnnCvtTensorInstr::wcnn_cvt_tensor_bf16_bf16;
    }

    static constexpr auto selected_cvt_tensor =
        WcnnCvtTensorType<GetCvtTensor<InDataType, AccDataType>(),
                          AuxData,
                          Clamp,
                          HPerWcnn,
                          WPerWcnn>{};

    __host__ __device__ constexpr WcnnCvtTensorSelector(){};
};

template <typename InDataType,
          typename AccDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t ActiveFun,
          bool Clamp,
          bool OutputChannelOffset = false>
struct WcnnCvtTensor
{
    __host__ __device__ constexpr WcnnCvtTensor(){};

    static constexpr index_t WaveSize = 32;

    //
    // Values:
    // shape_8x4x8 0x0
    // shape_4x4x8 0x1
    // shape_4x4x16 0x2
    // shape_4x2x16 0x3
    static constexpr index_t GetAuxData()
    {
        // int 8X4X8  = 0;
        // int 4X4X8  = 1;
        // int 4X4X16 = 2;
        // int 4X2X16 = 3;
        constexpr uint32_t Mod0 = []() {
            if constexpr((HPerWcnn == 8) && (WPerWcnn == 4))
                return 0;
            else if constexpr((HPerWcnn == 4) && (WPerWcnn == 4))
                return 2;
            else if constexpr((HPerWcnn == 4) && (WPerWcnn == 2))
                return 3;
            else
                static_assert("unsupport shape.");
        }();

        constexpr uint32_t Mod1 = (ActiveFun << 2) | (OutputChannelOffset ? (1 << 4) : 0);

        return Mod0 | (Mod1 << 6);
    };

    static constexpr auto AuxData = GetAuxData();

    static constexpr auto cvt_tensor_selector =
        WcnnCvtTensorSelector<InDataType, AccDataType, AuxData, Clamp, HPerWcnn, WPerWcnn>{};
    static constexpr auto cvt_tensor_instr = cvt_tensor_selector.selected_cvt_tensor;
};

} // namespace ck

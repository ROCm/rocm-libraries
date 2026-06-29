// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define EXECUTION_SPECIFIER __device__

typedef union cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;

    ushort ushortvec[2];
    float f32;
} cvt_bf16_fp32_t;

EXECUTION_SPECIFIER float bfloat16_to_float(ushort src_val)
{
    cvt_bf16_fp32_t target_val;

    target_val.ushortx2 = make_ushort2(0, src_val);

    return target_val.f32;
}

EXECUTION_SPECIFIER ushort float_to_bfloat16(float src_val)
{
    cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;
    // BF16 round and NaN preservation code matches
    // https://github.com/ROCm/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
        // mantissa bit. Quiet NaN is indicated by the most significant mantissa
        // bit being 1. Signaling NaN is indicated by the most significant
        // mantissa bit being 0 but some other bit(s) being 1. If any of the
        // lower 16 bits of the mantissa are 1, we set the least significant bit
        // of the bfloat16 mantissa, in order to preserve signaling NaN in case
        // the bloat16's mantissa bits are all 0.
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef HIP_PLUGIN_USE_RNE_BFLOAT16
        // When the exponent bits are not all 1s, then the value is zero, normal,
        // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
        // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
        // This causes the bfloat16's mantissa to be incremented by 1 if the 16
        // least significant bits of the float mantissa are greater than 0x8000,
        // or if they are equal to 0x8000 and the least significant bit of the
        // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
        // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
        // has the value 0x7f, then incrementing it causes it to become 0x00 and
        // the exponent is incremented by one, which is the next higher FP value
        // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
        // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
        // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
        // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
        // incrementing it causes it to become an exponent of 0xFF and a mantissa
        // of 0x00, which is Inf, the next higher value to the unrounded value.
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));

#endif // HIP_PLUGIN_USE_RNE_BFLOAT16
    }

    return target_val.ushortvec[1];
}

#ifdef __cplusplus
}
#endif

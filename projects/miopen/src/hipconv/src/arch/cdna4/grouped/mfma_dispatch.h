#pragma once

#include "packed_ops.h"
#include "types.h"

namespace hipconv::cdna4
{

__device__ inline fp32x4_t mfma_16x16x32(fp16x8_t a, fp16x8_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
}
__device__ inline fp32x4_t mfma_16x16x32(bf16x8_t a, bf16x8_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

__device__ inline fp32x4_t mfma_16x16x16(fp16x4_t a, fp16x4_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}
__device__ inline fp32x4_t mfma_16x16x16(bf16x4_t a, bf16x4_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ inline fp32x4_t mfma_4x4x4_16b(fp16x4_t a, fp16x4_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, c, 0, 0, 0);
}
__device__ inline fp32x4_t mfma_4x4x4_16b(bf16x4_t a, bf16x4_t b, fp32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c, 0, 0, 0);
}

// TF32 (simulated): inputs are fp32, computed via 3x BF16 MFMA.
// Caller is responsible for splitting each fp32x4 operand into a (big, small)
// bf16 pair via fp32x4_to_bf16_pair() — typically once per operand in the
// weight / input LDS->VGPR prologue, so the main loop reuses the prebuilt
// pair without recomputing the split. The wrapper expands to three BF16
// MFMAs (big*big + big*small + small*big); the 4th term (small*small) is
// dropped, contributing ~2^{-16} relative error per multiply. Inner
// mfma_4x4x4_16b calls dispatch to the existing BF16 overload above.
__device__ inline fp32x4_t mfma_4x4x4_16b(bf16_pair_x4 pa, bf16_pair_x4 pb, fp32x4_t c)
{
    c = mfma_4x4x4_16b(pa.big, pb.big, c);
    c = mfma_4x4x4_16b(pa.big, pb.small, c);
    c = mfma_4x4x4_16b(pa.small, pb.big, c);
    return c;
}

__device__ inline fp32x4_t mfma_16x16x16(bf16_pair_x4 pa, bf16_pair_x4 pb, fp32x4_t c)
{
    c = mfma_16x16x16(pa.big, pb.big, c);
    c = mfma_16x16x16(pa.big, pb.small, c);
    c = mfma_16x16x16(pa.small, pb.big, c);
    return c;
}

__device__ inline fp32x4_t mfma_16x16x32(bf16_pair_x8 pa, bf16_pair_x8 pb, fp32x4_t c)
{
    c = mfma_16x16x32(pa.big, pb.big, c);
    c = mfma_16x16x32(pa.big, pb.small, c);
    c = mfma_16x16x32(pa.small, pb.big, c);
    return c;
}

} // namespace hipconv::cdna4

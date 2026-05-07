// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Note: This header requires fp16x4_t and fp32x4_t to be visible
// (defined in types.hpp or via ck_tile/core.hpp). We avoid including
// types.hpp directly to prevent include-order issues with <hip/hip_fp8.h>.

namespace ck_tile::direct_conv
{

// MFMA functor for 4-channel kernel (mfma_f32_4x4x4f16).
//
// Lane mapping (64-lane wave, 16 independent 4x4 matmuls):
//   lane_col   = lane % 4  -> output column (N dimension)
//   lane_batch = lane / 4  -> batch index (0..15)
//
// Each lane provides 4 fp16 for A and B, receives 4 fp32 for C.
// C[4x4] += A[4xK] * B[Kx4] where K=4.
struct Mfma4x4x4
{
    using input_type = fp16x4_t;
    __device__ fp32x4_t operator()(fp16x4_t weight, fp16x4_t input, fp32x4_t acc) const
    {
        return __builtin_amdgcn_mfma_f32_4x4x4f16(weight, input, acc, 0, 0, 0);
    }
};

// MFMA functor for 16-channel kernel (mfma_f32_16x16x16f16).
//
// Lane mapping (64-lane wave, single 16x16 matmul):
//   lane_q  = lane % 16 -> output column (N dimension)
//   lane_c4 = lane / 16 -> K-reduction group (0..3)
//
// Each lane provides 4 fp16 for A and B from its reduction group,
// receives 4 fp32 for C. C[16x16] += A[16xK] * B[Kx16] where K=16.
struct Mfma16x16x16
{
    using input_type = fp16x4_t;
    __device__ fp32x4_t operator()(fp16x4_t weight, fp16x4_t input, fp32x4_t acc) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(weight, input, acc, 0, 0, 0);
    }
};

// MFMA functor for 8-channel kernel (mfma_f32_16x16x32_f16).
//
// Lane mapping (64-lane wave, single 16x16 matmul):
//   lane_q  = lane % 16 -> output column (N dimension)
//   lane_kg = lane / 16 -> K-reduction group (0..3)
//
// Each lane provides 8 fp16 for A and B from its reduction group,
// receives 4 fp32 for C. C[16x16] += A[16xK] * B[Kx16] where K=32.
//
// The 8c kernel uses Toeplitz structure: S-dimension is embedded in K=32
// as 4 filter taps x 8 channels, so no explicit S-loop is needed.
struct Mfma16x16x32
{
    // fp16x8_t is ck_tile::direct_conv::fp16x8_t (vector_size), matching InputLoaderToeplitz.
    using input_type = fp16x8_t;
    __device__ fp32x4_t operator()(fp16x8_t weight, fp16x8_t input, fp32x4_t acc) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_f16(weight, input, acc, 0, 0, 0);
    }
};

// MFMA functor for 32-channel kernel (mfma_f32_16x16x32_f16).
//
// Lane mapping (64-lane wave, single 16x16 matmul):
//   lane_q  = lane % 16 -> output column (N dimension)
//   lane_c8 = lane / 16 -> K-reduction group (0..3)
//
// Each lane provides 8 fp16 for A and B from its reduction group,
// receives 4 fp32 for C. C[16x16] += A[16xK] * B[Kx16] where K=32.
//
// Unlike the 8c Toeplitz variant (Mfma16x16x32), the 32c kernel uses
// full 32-channel reduction with explicit S-loop (INNER_KW = kw).
// Each group's 32 channels are split across 2 waves (wave_half).
struct Mfma16x16x32_32c
{
    using input_type = fp16x8_t;
    __device__ fp32x4_t operator()(fp16x8_t weight, fp16x8_t input, fp32x4_t acc) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_f16(weight, input, acc, 0, 0, 0);
    }
};

} // namespace ck_tile::direct_conv

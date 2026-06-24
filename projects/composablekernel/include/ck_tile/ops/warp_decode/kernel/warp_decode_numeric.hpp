// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct WarpDecodeNumeric
{
    CK_TILE_DEVICE static float dot2_bf16_packed_raw(float acc, uint32_t a, uint32_t b)
    {
        float dot = acc;
        // Match CK's dot2 asm pattern: tie the addend to the output and pad the dependent use.
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0\ns_nop 2"
                     : "=v"(dot)
                     : "v"(a), "v"(b), "0"(dot));
        return dot;
    }

    // s_nop-free v_dot2: only correct when the caller covers the write->read
    // hazard with independent work (e.g. several independent accumulators
    // interleaved) plus a dot2_drain4 before the results are read. Used by the
    // FP4 down path to avoid serializing the dot2 chain so the kernel stays
    // bandwidth-bound rather than compute-bound.
    CK_TILE_DEVICE static float dot2_bf16_packed_raw_nonop(float acc, uint32_t a, uint32_t b)
    {
        float dot = acc;
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0"
                     : "=v"(dot)
                     : "v"(a), "v"(b), "0"(dot));
        return dot;
    }

    // Cover the v_dot2 write latency once, just before reading the four
    // independent accumulators. The accumulators are tied as in/out so the
    // compiler cannot hoist the subsequent reads above the s_nop.
    CK_TILE_DEVICE static void dot2_drain4(float& a, float& b, float& c, float& d)
    {
#if defined(__gfx950__)
        asm volatile("s_nop 2" : "+v"(a), "+v"(b), "+v"(c), "+v"(d));
#else
        (void)a;
        (void)b;
        (void)c;
        (void)d;
#endif
    }

    CK_TILE_DEVICE static uint32_t pack_bf16_pair(bf16_t lo, bf16_t hi)
    {
        const uint32_t lo_bits = static_cast<uint32_t>(bit_cast<bf16_raw_t>(lo));
        const uint32_t hi_bits = static_cast<uint32_t>(bit_cast<bf16_raw_t>(hi));
        return lo_bits | (hi_bits << 16);
    }

    template <typename T>
    CK_TILE_DEVICE static bf16_t as_bf16_dot_operand(T x)
    {
        if constexpr(std::is_same_v<remove_cvref_t<T>, bf16_t>)
        {
            return x;
        }
        else
        {
            return type_convert<bf16_t>(type_convert<float>(x));
        }
    }

    template <typename ComputeDataType>
    CK_TILE_DEVICE static ComputeDataType dot2_bf16_scaled_add(
        ComputeDataType acc, bf16_t a0, bf16_t a1, bf16_t b0, bf16_t b1, ComputeDataType scale)
    {
        const uint32_t a = pack_bf16_pair(a0, a1);
        const uint32_t b = pack_bf16_pair(b0, b1);
        const float dot = dot2_bf16_packed_raw(0.0f, a, b);
        return acc + type_convert<ComputeDataType>(dot) * scale;
    }

    template <typename ComputeDataType>
    CK_TILE_DEVICE static ComputeDataType dot2_bf16_add(
        ComputeDataType acc, bf16_t a0, bf16_t a1, bf16_t b0, bf16_t b1)
    {
        const uint32_t a = pack_bf16_pair(a0, a1);
        const uint32_t b = pack_bf16_pair(b0, b1);
        const float dot = dot2_bf16_packed_raw(type_convert<float>(acc), a, b);
        return type_convert<ComputeDataType>(dot);
    }

    template <typename ComputeDataType>
    CK_TILE_DEVICE static ComputeDataType dot2_bf16_packed_lhs_add(
        ComputeDataType acc, uint32_t a, bf16_t b0, bf16_t b1)
    {
        const uint32_t b = pack_bf16_pair(b0, b1);
        const float dot = dot2_bf16_packed_raw(type_convert<float>(acc), a, b);
        return type_convert<ComputeDataType>(dot);
    }

    template <typename ComputeDataType>
    CK_TILE_DEVICE static ComputeDataType dot2_bf16_packed_add(
        ComputeDataType acc, uint32_t a, uint32_t b)
    {
        const float dot = dot2_bf16_packed_raw(type_convert<float>(acc), a, b);
        return type_convert<ComputeDataType>(dot);
    }

    template <index_t PairInWord>
    CK_TILE_DEVICE static uint32_t fp8x2_to_bf16x2(uint32_t fp8x4)
    {
        static_assert(PairInWord == 0 || PairInWord == 1);
#if defined(__gfx950__)
        union
        {
            bf16x2_t vec;
            uint32_t raw;
        } out;
        out.vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(
            fp8x4, type_convert<float>(1.0f), PairInWord);
        return out.raw;
#else
        (void)fp8x4;
        return 0;
#endif
    }

    // Convert one byte (two packed FP4 nibbles) of a 32-bit word holding eight
    // FP4 values into a BF16 pair (raw bits). ByteSel in [0,4) picks the byte;
    // scale applies the (MX) block scale during conversion. Mirrors
    // fp8x2_to_bf16x2 so the down-reduce dot2 loop is format-agnostic.
    template <index_t ByteSel>
    CK_TILE_DEVICE static uint32_t fp4x2_to_bf16x2(uint32_t fp4x8, float scale = 1.0f)
    {
        static_assert(ByteSel >= 0 && ByteSel < 4);
#if defined(__gfx950__)
        union
        {
            bf16x2_t vec;
            uint32_t raw;
        } out;
        out.vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(fp4x8, scale, ByteSel);
        return out.raw;
#else
        (void)fp4x8;
        (void)scale;
        return 0;
#endif
    }

    template <index_t PairInWord>
    CK_TILE_DEVICE static fp32x2_t fp8x2_to_f32x2(uint32_t fp8x4)
    {
        static_assert(PairInWord == 0 || PairInWord == 1);
#if defined(__gfx950__)
        return __builtin_amdgcn_cvt_pk_f32_fp8(fp8x4, PairInWord);
#else
        (void)fp8x4;
        return fp32x2_t{0.0f, 0.0f};
#endif
    }

    // Convert one byte (two packed FP4 nibbles) of a 32-bit word into an FP32
    // pair. ByteSel in [0,4) picks the byte; scale applies the (MX) block scale.
    // Pairs with v_pk_fma_f32 to give the FP4 down path an s_nop-free inner loop.
    template <index_t ByteSel>
    CK_TILE_DEVICE static fp32x2_t fp4x2_to_f32x2(uint32_t fp4x8, float scale = 1.0f)
    {
        static_assert(ByteSel >= 0 && ByteSel < 4);
#if defined(__gfx950__)
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(fp4x8, scale, ByteSel);
#else
        (void)fp4x8;
        (void)scale;
        return fp32x2_t{0.0f, 0.0f};
#endif
    }

    CK_TILE_DEVICE static fp32x2_t bf16x2_to_f32x2(uint32_t bf16x2)
    {
        const uint32_t lo = (bf16x2 & 0x0000ffffu) << 16;
        const uint32_t hi = bf16x2 & 0xffff0000u;
        return fp32x2_t{bit_cast<float>(lo), bit_cast<float>(hi)};
    }

    CK_TILE_DEVICE static fp32x2_t pk_fma_f32(fp32x2_t acc, fp32x2_t a, fp32x2_t b)
    {
#if defined(__gfx950__)
        fp32x2_t out;
        asm volatile("v_pk_fma_f32 %[out], %[a], %[b], %[acc]"
                     : [out] "=v"(out)
                     : [a] "v"(a), [b] "v"(b), [acc] "v"(acc));
        return out;
#else
        return acc + a * b;
#endif
    }

    CK_TILE_DEVICE static float horizontal_add(fp32x2_t v)
    {
        return v[0] + v[1];
    }
};

} // namespace ck_tile

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Numeric helpers for the warp-per-scalar dense GEMM kernels.
//
// P0 used only `wavefront_reduce_sum` (lifted from
// ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp:162-172) and the
// scalar atomic adders below. P0b adds the FP8/BF16 conversion and dot2 /
// pk_fma helpers, lifted verbatim from
// warp-decode-moe/.../warp_decode_numeric.hpp so the universal kernel can
// run an FP8 -> BF16 -> v_dot2_f32_bf16 chain (preferred on gfx950) or an
// FP8 -> FP32 -> v_pk_fma_f32 chain (alternate, kUsePackedFp32 = true).
struct GemmDecodeNumeric
{
    template <typename ComputeDataType>
    CK_TILE_DEVICE static ComputeDataType wavefront_reduce_sum(ComputeDataType val)
    {
        constexpr index_t num_stages = integer_log2_floor(get_warp_size());
        static_for<0, num_stages, 1>{}([&](auto istage) {
            const index_t offset       = 1 << istage.value;
            const index_t src_lane     = get_lane_id() ^ offset;
            const ComputeDataType peer = warp_shuffle(val, src_lane);
            val += peer;
        });
        return val;
    }

    CK_TILE_DEVICE static float dot2_bf16_packed_raw(float acc, uint32_t a, uint32_t b)
    {
        float dot = acc;
        // Match CK's dot2 asm pattern: tie the addend to the output and pad the dependent use.
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0\ns_nop 2"
                     : "=v"(dot)
                     : "v"(a), "v"(b), "0"(dot));
        return dot;
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
    CK_TILE_DEVICE static ComputeDataType
    dot2_bf16_packed_add(ComputeDataType acc, uint32_t a, uint32_t b)
    {
        const float dot = dot2_bf16_packed_raw(type_convert<float>(acc), a, b);
        return type_convert<ComputeDataType>(dot);
    }

    // Convert one FP8x2 pair (PairInWord = 0 -> low half, 1 -> high half) of an
    // FP8x4 word to a packed BF16x2 word using the gfx950 cvt builtin. Used by
    // the FP8 dot2 K-loop body in the universal and blockscale kernels.
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

    // Convert one FP8x2 pair to fp32x2 via the gfx950 cvt builtin. Used by the
    // alternate kUsePackedFp32 = true path.
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

    CK_TILE_DEVICE static float horizontal_add(fp32x2_t v) { return v[0] + v[1]; }
};

// Scalar atomic-add helpers used by the AtomicAdd split-K epilogue.
//
// Each warp writes one C scalar (kNPerWarp = 1, kMPerWarp = 1 in P0); the CK
// Tile core only provides paired atomic_add for bf16x2_t / fp16x2_t (see
// core/arch/generic_memory_space_atomic.hpp). We reuse those by widening the
// scalar to a pair where the unused half is zero, addressing the natural
// 4-byte slot containing the target element. The 32-bit CAS loop inside
// atomic_add<bf16x2_t> handles concurrent writes to the same slot from
// adjacent warps (n and n+1).
//
// Constraints surfaced through GemmDecodeUniversalKernel::IsSupportedArgument
// when k_batch > 1:
//   - C buffer must be 4-byte aligned (hipMalloc guarantees this).
//   - For non-divisible N, the helper may touch the immediately adjacent
//     half of the 4-byte slot. That slot is part of the same row of C, so
//     no out-of-bounds occurs except when N is odd and the warp targets the
//     last column. Tests in P0 use even N exclusively.

CK_TILE_DEVICE void gemm_decode_atomic_add(bf16_t* p_dst, bf16_t x)
{
    const auto addr      = reinterpret_cast<uintptr_t>(p_dst);
    const bool is_hi     = (addr & 0x2u) != 0;
    auto* p32            = reinterpret_cast<bf16x2_t*>(addr & ~uintptr_t(0x3u));
    const bf16_t zero_bf = type_convert<bf16_t>(0.0f);
    bf16x2_t add_pair;
    add_pair[0] = is_hi ? zero_bf : x;
    add_pair[1] = is_hi ? x : zero_bf;
    atomic_add<bf16x2_t>(p32, add_pair);
}

CK_TILE_DEVICE void gemm_decode_atomic_add(fp16_t* p_dst, fp16_t x)
{
    const auto addr       = reinterpret_cast<uintptr_t>(p_dst);
    const bool is_hi      = (addr & 0x2u) != 0;
    auto* p32             = reinterpret_cast<fp16x2_t*>(addr & ~uintptr_t(0x3u));
    const fp16_t zero_fp  = type_convert<fp16_t>(0.0f);
    fp16x2_t add_pair;
    add_pair[0] = is_hi ? zero_fp : x;
    add_pair[1] = is_hi ? x : zero_fp;
    atomic_add<fp16x2_t>(p32, add_pair);
}

CK_TILE_DEVICE void gemm_decode_atomic_add(float* p_dst, float x)
{
    atomicAdd(p_dst, x);
}

} // namespace ck_tile

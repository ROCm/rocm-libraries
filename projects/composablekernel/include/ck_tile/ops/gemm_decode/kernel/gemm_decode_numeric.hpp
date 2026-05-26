// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Numeric helpers for the warp-per-scalar dense GEMM kernels. P0 uses only
// `wavefront_reduce_sum` (lifted unchanged from
// ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp:162-172) and the
// scalar atomic adders below for the AtomicAdd split-K epilogue. The dot2 /
// fp8x2 / pk_fma helpers from warp_decode_numeric.hpp are intentionally not
// pulled in until the FP8 / blockscale / MX paths land in P0b/P1/P2.
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

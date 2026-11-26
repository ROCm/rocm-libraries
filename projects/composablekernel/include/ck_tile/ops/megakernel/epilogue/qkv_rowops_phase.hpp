// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/epilogue/epilogue_common.hpp"

namespace ck_tile {

/// @file
/// The qkv "row ops" of the target model: what follows the fused q/k/v/gate projection.
///
/// Fused GEMM output of one TP rank, one bf16 row per token (columns, 64 per head):
///   [ q: kQHeads heads | k: kKvHeads heads | v: kKvHeads heads | gate: kQHeads heads ]
/// TP8: 16 + 2 + 2 + 16 heads = 2304 columns. Every section starts at a multiple of 64.
///
/// Per token row:
///   q    = bf16(bf16(RMSNorm(q_head, 1e-5)) * 5.4730064863875)   -> RoPE if sliding -> q_out
///   k    = bf16(RMSNorm(k_head, 1e-5))                           -> RoPE if sliding -> K cache
///   v    = v_head                                                                    -> V cache
///   gate : left in place; the attention gate reads it at column kGateCol.
/// RoPE (rotate_half): out[d] = bf16(bf16(x[d] * c) + bf16(rot[d] * s)), rot = cat(-x[32:],
/// x[:32]), c / s = bf16(cos / sin(pos * inv_freq[d % 32])), inv_freq[i] = 1 / 5e5^(2i / 64).
/// Global layers (NoPE) skip it. These are HF's bf16 rounding points (its cos / sin are bf16).
///
/// Paged KV cache (one per K and V, same layout):
///   cache[num_pages][kKvHeads][kPageSize][64], KvDataType = bf16_t or fp8_t (e4m3fn).
/// Token row r goes to slot = slot_mapping[r] = page * kPageSize + offset; slot < 0 skips the row
/// (padding). fp8 stores x / scale (per-tensor k_scale, v_scale); the reader multiplies back.
/// A (page, kv head) block is kPageSize x 64 contiguous elements, one TDM 2D tile.
///
/// Why a phase and not a GEMM epilogue: the norms need a whole 64-wide head inside one output
/// tile, which holds for any NPerBlock that is a multiple of 64 (tile origins are then 64-aligned
/// and no head straddles tiles), so an epilogue is possible. But TdmEpilogue / CShuffleEpilogue
/// only offer an element-wise functor; a row functor needs a hook on the staged LDS C tile
/// (between its cast to LDS and the TDM store), a shared-code change. The GEMM output is
/// M x 2304 bf16 (<= 295 KB at M = 64, L2 resident), so a separate phase costs one extra L2 read
/// and a dependency wait. MkQkHeadOp is the per-head device op that such a hook would call.
///
/// Work item: kHeadsPerItem = 4 kNumWarps consecutive (row, head) pairs of the
/// M x (kQHeads + 2 kKvHeads) "head rows"; 8 lanes per head row, 8 elements per lane (a wave32
/// holds 4 head rows; on the host, items are counted the same way).
/// One counter slot covers all rows (the attention phase needs every row of the step anyway).

/// Per-head ops on one 64-wide head row held by 8 consecutive lanes, 8 elements per lane
/// (lane l of the group holds elements [8l, 8l + 8)). All 8 lanes must be active.
struct MkQkHeadOp
{
    static constexpr index_t kLanes = 8;
    static constexpr index_t kVec   = 8;

    /// x <- bf16(RMSNorm(x, eps)) [then bf16(x * scale) if kScale].
    template <bool kScale>
    CK_TILE_DEVICE static void Norm(float (&x)[kVec], float scale)
    {
        float ss = 0.f;
        static_for<0, kVec, 1>{}([&](auto i) { ss += x[i.value] * x[i.value]; });
        ss            = mk_group_sum<kLanes>(ss);
        const float r = rsqrtf(ss * (1.f / MkConstants::kHeadDim) + MkConstants::kRmsEps);
        static_for<0, kVec, 1>{}([&](auto i) {
            float y = mk_round_bf16(x[i.value] * r);
            if constexpr(kScale)
                y = mk_round_bf16(y * scale);
            x[i.value] = y;
        });
    }

    /// rotate_half RoPE; @p cos_sin is the table row of this token ([0, 32) cos, [32, 64) sin).
    CK_TILE_DEVICE static void Rope(float (&x)[kVec], const float* cos_sin)
    {
        const index_t l = get_lane_id() % kLanes;
        // Partner lane holds element d +- 32; x is bf16-exact, so pack pairs for the shuffle.
        float p[kVec];
        static_for<0, kVec / 2, 1>{}([&](auto i) {
            const uint32_t lo  = bit_cast<uint32_t>(x[2 * i.value]) >> 16;
            const uint32_t hi  = bit_cast<uint32_t>(x[2 * i.value + 1]) & 0xFFFF0000u;
            const uint32_t v   = warp_shuffle(lo | hi, static_cast<uint32_t>(get_lane_id() ^ 4));
            p[2 * i.value]     = bit_cast<float>(v << 16);
            p[2 * i.value + 1] = bit_cast<float>(v & 0xFFFF0000u);
        });
        const float sign = l < 4 ? -1.f : 1.f;
        float c[kVec], s[kVec];
        mk_load_f32x8(cos_sin + (l % 4) * kVec, c);
        mk_load_f32x8(cos_sin + 32 + (l % 4) * kVec, s);
        static_for<0, kVec, 1>{}([&](auto i) {
            const float xc = mk_round_bf16(x[i.value] * mk_round_bf16(c[i.value]));
            const float rs = mk_round_bf16(sign * p[i.value] * mk_round_bf16(s[i.value]));
            x[i.value]     = mk_round_bf16(xc + rs);
        });
    }
};

/// Host: the RoPE table HF builds (fp32 inv_freq, fp32 angle), rows [0, max_pos), 64 floats per
/// row: cos(pos * inv_freq[i]) for i < 32, then sin.
CK_TILE_HOST std::vector<float> mk_make_rope_table(index_t max_pos,
                                                     float theta = MkConstants::kRopeTheta)
{
    std::vector<float> t(static_cast<size_t>(max_pos) * 64);
    for(index_t i = 0; i < 32; ++i)
    {
        const float inv_freq = 1.f / std::pow(theta, static_cast<float>(2 * i) / 64.f);
        for(index_t pos = 0; pos < max_pos; ++pos)
        {
            const float f                             = static_cast<float>(pos) * inv_freq;
            t[static_cast<size_t>(pos) * 64 + i]      = std::cos(f);
            t[static_cast<size_t>(pos) * 64 + 32 + i] = std::sin(f);
        }
    }
    return t;
}

template <index_t kNumWarps_,
          typename KvDataType_ = bf16_t,
          index_t kPageSize_   = 16,
          index_t kQHeads_     = 16,
          index_t kKvHeads_    = 2>
struct MkQkvRowOpsPhase
{
    using KvDataType = remove_cvref_t<KvDataType_>;
    static_assert(std::is_same_v<KvDataType, bf16_t> || std::is_same_v<KvDataType, fp8_t>,
                  "KV cache is bf16 or fp8 e4m3");

    /// As GemmPhase: waves times the wave size of the compiler pass (host passes assume 64).
    static constexpr index_t kBlockSize = kNumWarps_ * get_warp_size();
    static constexpr index_t kPageSize  = kPageSize_;
    static constexpr index_t kQHeads    = kQHeads_;
    static constexpr index_t kKvHeads   = kKvHeads_;
    static constexpr index_t kHeadDim   = MkConstants::kHeadDim;

    static constexpr index_t kQCol    = 0;
    static constexpr index_t kKCol    = kQHeads * kHeadDim;
    static constexpr index_t kVCol    = kKCol + kKvHeads * kHeadDim;
    static constexpr index_t kGateCol = kVCol + kKvHeads * kHeadDim;
    static constexpr index_t kN       = kGateCol + kQHeads * kHeadDim;

    static constexpr index_t kHeadRowsPerToken = kQHeads + 2 * kKvHeads;
    static constexpr index_t kHeadsPerItem     = 4 * kNumWarps_;
    /// One slot for all rows; large enough that WaitDependency's divisions stay in range.
    static constexpr index_t kSlotRows = 1 << 24;

    using barrier_pipeline = void;

    struct Args
    {
        const bf16_t* qkvg;          ///< [M, qkvg_stride], columns as in @file
        index_t qkvg_stride;         ///< elements, multiple of 8
        bf16_t* q_out;               ///< [M, q_stride], kQHeads * 64 used
        index_t q_stride;            ///< elements, multiple of 8
        KvDataType* k_cache;         ///< [num_pages][kKvHeads][kPageSize][64]
        KvDataType* v_cache;         ///< same layout
        const int32_t* slot_mapping; ///< [M], page * kPageSize + offset, < 0 = skip
        const int32_t* positions;    ///< [M], RoPE position of each row
        const float* rope_table;     ///< mk_make_rope_table layout; unused when !use_rope
        float k_scale;               ///< fp8 cache only: stored = k / k_scale
        float v_scale;
        index_t M;
        index_t use_rope; ///< 1: sliding layer (RoPE), 0: global layer (NoPE)
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST_DEVICE static index_t NumHeadRows(const Args& a)
    {
        return a.M * kHeadRowsPerToken;
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a)
    {
        return integer_divide_ceil(NumHeadRows(a), kHeadsPerItem);
    }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t) { return NumWork(a); }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }

    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t h0 = work * kHeadsPerItem;
        const index_t h1 = min(NumHeadRows(a), h0 + kHeadsPerItem);
        return make_tuple(h0 / kHeadRowsPerToken, (h1 - 1) / kHeadRowsPerToken + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.M > 0 && a.qkvg_stride >= kN && a.qkvg_stride % 8 == 0 && a.q_stride >= kKCol &&
               a.q_stride % 8 == 0 && a.qkvg != nullptr && a.q_out != nullptr &&
               a.k_cache != nullptr && a.v_cache != nullptr && a.slot_mapping != nullptr &&
               (a.use_rope == 0 || (a.positions != nullptr && a.rope_table != nullptr));
    }

    CK_TILE_DEVICE static auto InitBarriers() {}

    CK_TILE_DEVICE static void StoreKv(KvDataType* p, const float (&x)[8], float inv_scale)
    {
        if constexpr(std::is_same_v<KvDataType, bf16_t>)
        {
            ignore = inv_scale;
            mk_store_bf16x8(p, x);
        }
        else
        {
            uint32x2_t v;
            v[0] = mk_pack_fp8x4(
                x[0] * inv_scale, x[1] * inv_scale, x[2] * inv_scale, x[3] * inv_scale);
            v[1] = mk_pack_fp8x4(
                x[4] * inv_scale, x[5] * inv_scale, x[6] * inv_scale, x[7] * inv_scale);
            *c_style_pointer_cast<uint32x2_t*>(p) = v;
        }
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* /*smem*/, const Token&)
    {
        const index_t group = get_thread_id() / MkQkHeadOp::kLanes;
        const index_t lane  = get_thread_id() % MkQkHeadOp::kLanes;
        const index_t hr    = work * kHeadsPerItem + group;
        if(group >= kHeadsPerItem || hr >= NumHeadRows(a))
            return; // whole 8-lane groups drop out together
        const index_t row  = hr / kHeadRowsPerToken;
        const index_t head = hr - row * kHeadRowsPerToken;                // q heads, then k, then v
        const index_t col  = head * kHeadDim + lane * MkQkHeadOp::kVec; // sections are adjacent

        float x[8];
        mk_load_bf16x8(a.qkvg + static_cast<long_index_t>(row) * a.qkvg_stride + col, x);

        const bool is_q = head < kQHeads;
        const bool is_v = head >= kQHeads + kKvHeads;
        if(!is_v)
        {
            if(is_q)
                MkQkHeadOp::Norm<true>(x, MkConstants::kQkScale);
            else
                MkQkHeadOp::Norm<false>(x, 1.f);
            if(a.use_rope != 0)
            {
                const index_t pos = a.positions[row];
                MkQkHeadOp::Rope(x, a.rope_table + static_cast<long_index_t>(pos) * kHeadDim);
            }
        }

        if(is_q)
        {
            mk_store_bf16x8(a.q_out + static_cast<long_index_t>(row) * a.q_stride + col, x);
            return;
        }
        const index_t slot = a.slot_mapping[row];
        if(slot < 0)
            return;
        const index_t page   = slot / kPageSize;
        const index_t offset = slot - page * kPageSize;
        const index_t kv     = is_v ? head - kQHeads - kKvHeads : head - kQHeads;
        const long_index_t dst =
            ((static_cast<long_index_t>(page) * kKvHeads + kv) * kPageSize + offset) * kHeadDim +
            lane * MkQkHeadOp::kVec;
        if(is_v)
            StoreKv(a.v_cache + dst, x, 1.f / a.v_scale);
        else
            StoreKv(a.k_cache + dst, x, 1.f / a.k_scale);
    }
};

} // namespace ck_tile

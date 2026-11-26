// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/epilogue/epilogue_common.hpp"

namespace ck_tile {

/// @file
/// Attention output gate of the target model, per q head of 64:
///   o   = bf16(attention output)                        (HF's attention returns bf16)
///   o   = bf16(RMSNorm(o, 1e-5))                        (o_norm, no weight)
///   out = bf16(o * bf16(sigmoid(gate)))                 (gate = gate_proj pre-activation, bf16)
/// out is the o_proj input [M, kQHeads * 64].
///
/// MkAttnOutGate is the device op for the attention phase's combine step (after the split-KV
/// partials of a head row are merged and divided by the softmax sum); MkAttnGatePhase is the
/// same op as a standalone phase over a bf16 attention output.

/// One 64-wide head row held by kLanes consecutive lanes, 64 / kLanes consecutive elements each
/// (lane l holds [l * kVec, (l + 1) * kVec)). All kLanes lanes must be active.
template <index_t kLanes_>
struct MkAttnOutGate
{
    static constexpr index_t kLanes = kLanes_;
    static constexpr index_t kVec   = MkConstants::kHeadDim / kLanes;
    static_assert(kLanes >= 1 && kLanes <= 32 && MkConstants::kHeadDim % kLanes == 0);

    /// o: attention output (fp32), gate: gate pre-activation (bf16 values); o becomes the
    /// bf16-exact o_proj input.
    CK_TILE_DEVICE static void Apply(float (&o)[kVec], const float (&gate)[kVec])
    {
        float ss = 0.f;
        static_for<0, kVec, 1>{}([&](auto i) {
            o[i.value] = mk_round_bf16(o[i.value]);
            ss += o[i.value] * o[i.value];
        });
        ss            = mk_group_sum<kLanes>(ss);
        const float r = rsqrtf(ss * (1.f / MkConstants::kHeadDim) + MkConstants::kRmsEps);
        static_for<0, kVec, 1>{}([&](auto i) {
            const float n = mk_round_bf16(o[i.value] * r);
            const float g = mk_round_bf16(mk_sigmoid(gate[i.value]));
            o[i.value]    = mk_round_bf16(n * g);
        });
    }
};

/// Standalone phase: work item = 4 kNumWarps consecutive (row, head) pairs, 8 lanes each.
/// One counter slot for all rows.
template <index_t kNumWarps_, index_t kQHeads_ = 16>
struct MkAttnGatePhase
{
    /// As GemmPhase: waves times the wave size of the compiler pass (host passes assume 64).
    static constexpr index_t kBlockSize    = kNumWarps_ * get_warp_size();
    static constexpr index_t kQHeads       = kQHeads_;
    static constexpr index_t kLanes        = 8;
    static constexpr index_t kHeadsPerItem = 4 * kNumWarps_;
    static constexpr index_t kSlotRows     = 1 << 24;
    using Op                               = MkAttnOutGate<kLanes>;
    using barrier_pipeline                 = void;

    struct Args
    {
        const bf16_t* attn;  ///< [M, attn_stride], kQHeads * 64 used
        index_t attn_stride; ///< elements, multiple of 8
        const bf16_t* gate;  ///< [M, gate_stride]; e.g. qkvg + MkQkvRowOpsPhase::kGateCol
        index_t gate_stride;
        bf16_t* out; ///< [M, out_stride], o_proj input
        index_t out_stride;
        index_t M;
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }
    CK_TILE_HOST_DEVICE static index_t NumHeadRows(const Args& a) { return a.M * kQHeads; }
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
        return make_tuple(h0 / kQHeads, (h1 - 1) / kQHeads + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const index_t n = kQHeads * MkConstants::kHeadDim;
        return a.M > 0 && a.attn && a.gate && a.out && a.attn_stride >= n &&
               a.attn_stride % 8 == 0 && a.gate_stride >= n && a.gate_stride % 8 == 0 &&
               a.out_stride >= n && a.out_stride % 8 == 0;
    }

    CK_TILE_DEVICE static auto InitBarriers() {}

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* /*smem*/, const Token&)
    {
        const index_t group = get_thread_id() / kLanes;
        const index_t hr    = work * kHeadsPerItem + group;
        if(group >= kHeadsPerItem || hr >= NumHeadRows(a))
            return;
        const index_t row = hr / kQHeads;
        const index_t col =
            (hr - row * kQHeads) * MkConstants::kHeadDim + (get_thread_id() % kLanes) * 8;
        float o[8], g[8];
        mk_load_bf16x8(a.attn + static_cast<long_index_t>(row) * a.attn_stride + col, o);
        mk_load_bf16x8(a.gate + static_cast<long_index_t>(row) * a.gate_stride + col, g);
        Op::Apply(o, g);
        mk_store_bf16x8(a.out + static_cast<long_index_t>(row) * a.out_stride + col, o);
    }
};

} // namespace ck_tile

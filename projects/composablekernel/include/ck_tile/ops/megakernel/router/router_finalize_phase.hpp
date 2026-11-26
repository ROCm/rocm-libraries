// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/moe/route_table.hpp"

namespace ck_tile {

namespace impl {
/// Total order on (choice, expert): larger choice first, then the smaller expert index.
/// Keys of fp32 values compare like the values except -0 < +0 and NaN above +inf.
CK_TILE_DEVICE uint64_t router_key(float choice, index_t expert)
{
    const uint32_t b = bit_cast<uint32_t>(choice);
    const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<uint64_t>(o) << 32) | (0xffffffffu - static_cast<uint32_t>(expert));
}

/// v of lane (lane ^ Mask), wave32. DPP row_xmask within a row of 16, permlanex16 across rows.
template <index_t Mask>
CK_TILE_DEVICE uint32_t router_xor_shuffle(uint32_t v)
{
    const int x = bit_cast<int>(v);
    if constexpr(Mask == 16)
        return bit_cast<uint32_t>(
            __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
    else
        return bit_cast<uint32_t>(__builtin_amdgcn_update_dpp(x, x, 0x160 | Mask, 0xf, 0xf, false));
}
} // namespace impl

/// @brief MoE router finalize as one megakernel phase: per row, sigmoid of the router logits,
/// top-k of (sigmoid + e_score_correction_bias), weights = sigmoid at the chosen experts divided
/// by (their sum + eps) times routed_scaling_factor; then the whole moe_route_table.
///
/// One work item: one workgroup owns the table (M <= kMaxM <= 64). A wave owns a row; lane l
/// holds experts l, l + 32, ... Top-k is kTopK rounds of a wave arg-max on a 64-bit key.
///
/// Ties: on bitwise-equal fp32 choice the smaller expert index wins. Slots are in selection
/// order: descending choice, ties by ascending expert. The weight sum adds the slots in order.
///
/// Every table field a reader may use is rewritten each launch: topk_ids / topk_weights of rows
/// < M, expert_count of all experts, expert_routes[e * max_m + i] for i < expert_count[e],
/// active (all num_experts entries: the active experts ascending, then the idle ones ascending)
/// and num_active. No reset between launches.
///
/// Input: logits [M, ld_logits] fp32 (already scaled), written by RouterGemmPhase (after_all).
template <index_t kBlockSize_  = 256,
          index_t kMaxM_       = 64,
          index_t kNumExperts_ = 256,
          index_t kTopK_       = 8>
struct RouterFinalizePhase
{
    /// Threads of the workgroup on the device (wave32). kBlockSize is the same wave count in
    /// the lanes of the compile pass (the host pass sees 64-lane waves), as Megakernel expects;
    /// it must not feed the device code, which uses kThreads.
    static constexpr index_t kThreads = kBlockSize_;
    static constexpr index_t kWaveSize   = 32;
    static constexpr index_t kWaves      = kThreads / kWaveSize;
    static constexpr index_t kBlockSize  = kWaves * get_warp_size();
    static constexpr index_t kMaxM       = kMaxM_;
    static constexpr index_t kNumExperts = kNumExperts_;
    static constexpr index_t kTopK       = kTopK_;
    static constexpr index_t kPerLane    = kNumExperts / kWaveSize;
    static constexpr index_t kPerThread  = kNumExperts / kThreads;
    static constexpr index_t kSlotRows   = kMaxM;
    static constexpr uint8_t kNoSlot     = 0xff;

    static_assert(kThreads % kWaveSize == 0, "wave32 workgroup");
    static_assert(kNumExperts % kThreads == 0 && kNumExperts % kWaveSize == 0, "expert split");
    static_assert(kPerLane <= 32, "a lane tracks its taken experts in 32 bits");
    static_assert(kTopK <= kWaveSize && kTopK < 255, "top-k too large");
    static_assert(kMaxM <= 64 && kMaxM % 16 == 0, "kMaxM in 16..64");

    using barrier_pipeline = void;

    struct Args
    {
        const float* logits; // [M, ld_logits]
        index_t ld_logits;
        const float* bias; // e_score_correction_bias [kNumExperts]
        float routed_scaling_factor;
        float eps; // 1e-15
        moe_route_table table;
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // slot[e][row]: slot of (row, e) or kNoSlot; then the per-wave scan totals.
        return kNumExperts * kMaxM + kWaves * sizeof(index_t);
    }

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t)
    {
        return make_tuple(index_t{0}, a.table.M);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const auto& t = a.table;
        return t.M >= 1 && t.M <= kMaxM && t.max_m >= t.M && t.top_k == kTopK &&
               t.num_experts == kNumExperts && a.ld_logits >= kNumExperts && a.logits != nullptr &&
               a.bias != nullptr && t.topk_ids != nullptr && t.topk_weights != nullptr &&
               t.expert_count != nullptr && t.expert_routes != nullptr && t.active != nullptr &&
               t.num_active != nullptr;
    }

    CK_TILE_DEVICE static auto InitBarriers() { return 0; }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t, void* smem, const Token&)
    {
        const moe_route_table& t = a.table;
        uint8_t* slot            = static_cast<uint8_t*>(smem);
        index_t* wave_sum        = reinterpret_cast<index_t*>(slot + kNumExperts * kMaxM);
        const index_t M          = t.M;
        const index_t wave       = get_warp_id();
        const index_t lane       = get_lane_id();

        for(index_t i = threadIdx.x; i < kNumExperts * kMaxM / 4; i += kThreads)
            reinterpret_cast<uint32_t*>(slot)[i] = 0xffffffffu;
        block_sync_lds();

        // TODO: rows are independent; with M < kWaves some waves idle. Per-lane presort of the
        // kPerLane keys would cut each round's local scan if this phase shows on the timeline.
        for(index_t row = wave; row < M; row += kWaves)
        {
            float s[kPerLane];
            uint64_t key[kPerLane];
            static_for<0, kPerLane, 1>{}([&](auto j) {
                const index_t e = lane + j * kWaveSize;
                const float x   = a.logits[row * a.ld_logits + e];
                s[j]            = 1.f / (1.f + expf(-x));
                key[j]          = impl::router_key(s[j] + a.bias[e], e);
            });

            float sum      = 0.f;
            index_t my_id  = 0;
            float my_s     = 0.f;
            uint32_t taken = 0;
            static_for<0, kTopK, 1>{}([&](auto k) {
                uint64_t best = 0; // below every real key
                static_for<0, kPerLane, 1>{}([&](auto j) {
                    const uint64_t kj = (taken >> j) & 1u ? uint64_t{0} : key[j];
                    best              = kj > best ? kj : best;
                });
                static_for<0, 5, 1>{}([&](auto st) {
                    constexpr index_t mask = 16 >> st;
                    const uint32_t lo = impl::router_xor_shuffle<mask>(static_cast<uint32_t>(best));
                    const uint32_t hi =
                        impl::router_xor_shuffle<mask>(static_cast<uint32_t>(best >> 32));
                    const uint64_t o = (static_cast<uint64_t>(hi) << 32) | lo;
                    best             = o > best ? o : best;
                });
                const index_t e = static_cast<index_t>(0xffffffffu - static_cast<uint32_t>(best));
                const index_t owner = e % kWaveSize;
                const index_t je    = e / kWaveSize;
                float se            = 0.f;
                static_for<0, kPerLane, 1>{}([&](auto j) { se = j == je ? s[j] : se; });
                if(lane == owner)
                    taken |= 1u << je;
                se = bit_cast<float>(__builtin_amdgcn_readlane(bit_cast<int>(se), owner));
                sum += se;
                if(lane == k)
                {
                    my_id = e;
                    my_s  = se;
                }
            });

            if(lane < kTopK)
            {
                const index_t route       = row * kTopK + lane;
                t.topk_ids[route]         = my_id;
                t.topk_weights[route]     = my_s / (sum + a.eps) * a.routed_scaling_factor;
                slot[my_id * kMaxM + row] = static_cast<uint8_t>(lane);
            }
        }
        block_sync_lds();

        // Per expert: routes ascending (= rows ascending), count, then the active list by scan.
        index_t cnt[kPerThread];
        index_t local = 0;
        static_for<0, kPerThread, 1>{}([&](auto j) {
            const index_t e = threadIdx.x * kPerThread + j;
            index_t c       = 0;
            for(index_t row = 0; row < M; ++row)
            {
                const uint8_t sl = slot[e * kMaxM + row];
                if(sl != kNoSlot)
                {
                    t.expert_routes[e * t.max_m + c] = row * kTopK + sl;
                    ++c;
                }
            }
            t.expert_count[e] = c;
            cnt[j]            = c;
            local += c > 0 ? 1 : 0;
        });

        // Exclusive scan of the active flags over threads (expert order).
        index_t incl = local;
        static_for<0, 5, 1>{}([&](auto st) {
            constexpr index_t d = 1 << st;
            const index_t y     = warp_shuffle(incl, static_cast<uint32_t>(max(lane - d, 0)));
            if(lane >= d)
                incl += y;
        });
        if(lane == kWaveSize - 1)
            wave_sum[wave] = incl;
        block_sync_lds();
        index_t pos = incl - local, total = 0;
        static_for<0, kWaves, 1>{}([&](auto w) {
            const index_t v = wave_sum[w];
            pos += w < wave ? v : 0;
            total += v;
        });

        static_for<0, kPerThread, 1>{}([&](auto j) {
            const index_t e = threadIdx.x * kPerThread + j;
            if(cnt[j] > 0)
                t.active[pos++] = e;
            else
                t.active[total + e - pos] = e; // idle experts after the active ones
        });
        if(threadIdx.x == 0)
            *t.num_active = total;
    }
};

} // namespace ck_tile

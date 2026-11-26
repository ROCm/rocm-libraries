// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_highprec_traits.hpp"

namespace ck_tile {

namespace impl {
/// One 16-byte global load from a 16-byte aligned address.
template <typename V>
CK_TILE_DEVICE V router_load16(const void* p)
{
    static_assert(sizeof(V) == 16, "16-byte vector");
    V v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 16), 16);
    return v;
}
} // namespace impl

/// @brief MoE router logits as one megakernel phase (gfx1250):
///   logits[m, n] = scale * sum_k float(h[m, k]) * w[n, k]
/// with bf16 h [M, K] (row stride stride_h), fp32 w [num_experts, K] (K contiguous), fp32 logits
/// [M, ld_logits]. M <= kMaxM rows; at decode M the phase streams w once (weight-bandwidth bound).
///
/// Math: v_wmma_f32_16x16x4_f32 (fp32 WMMA; A = 16 rows of h, B = 16 experts of w^T). A lane loads
/// 8 contiguous k of one h row and of one w row and feeds 4 WMMAs: WMMA i takes k {2i, 2i+1} of
/// lane half 0 and k {8 + 2i, 8 + 2i + 1} of lane half 1, the same permutation on A and B.
///
/// Work: item = (split, n tile); an n tile is kExpertsPerItem experts of this rank's shard, a split
/// is a contiguous K range (in units of 16). Within an item the waves interleave over the K units,
/// every wave covering all kExpertsPerItem experts, then add up through LDS in wave order.
///
/// Split-K reduction (k_splits > 1): each item writes its sum to partial[split][n_tile][m][e] and
/// takes a ticket on tickets[n_tile]. The last of the k_splits items of an n tile adds the
/// partials in split order (deterministic), scales, and writes logits; it also resets the ticket
/// to 0 for the next launch. So the logits need no zeroing, and partial / tickets are scratch:
/// tickets is zeroed once at allocation and k_splits must stay the same for every launch that
/// shares it. With k_splits == 1 items write logits directly.
///
/// Tensor parallel: rank / world shard the experts (N). The phase writes only columns
/// [rank * E / world, (rank + 1) * E / world) of logits; gathering the other columns is left to
/// the collectives (item 2). world = 1 computes every expert.
///
/// Counters: one slot covering all rows, so a consumer can only wait on after_all.
///
/// Sizing: with k_splits = 12 on 96 workgroups an item is 43 K units, at most 6 per wave, so the
/// default kUnroll = 6 issues all of a wave's loads at once. VGPRs (gfx1250, no spills):
/// kMaxM 16 / 64 -> 114 / 216 at kUnroll 4, 154 / 280 at 6, 194 / 344 at 8.
/// TODO: software-pipeline the unroll groups (the compute waits for loadcnt 0 of its group).
template <index_t kBlockSize_     = 256,
          index_t kMaxM_          = 64,
          index_t kNTilesPerWave_ = 2,
          index_t kUnroll_        = 6>
struct RouterGemmPhase
{
    /// Threads of the workgroup on the device (wave32). kBlockSize is the same wave count in
    /// the lanes of the compile pass (the host pass sees 64-lane waves), as Megakernel expects;
    /// it must not feed the device code, which uses kThreads.
    static constexpr index_t kThreads = kBlockSize_;
    static constexpr index_t kWaveSize       = 32;
    static constexpr index_t kWaves          = kThreads / kWaveSize;
    static constexpr index_t kBlockSize      = kWaves * get_warp_size();
    static constexpr index_t kMaxM           = kMaxM_;
    static constexpr index_t kMTiles         = kMaxM / 16;
    static constexpr index_t kNTiles         = kNTilesPerWave_;
    static constexpr index_t kExpertsPerItem = kNTiles * 16;
    /// K per unit: one 8-element load per lane half, 4 WMMAs.
    static constexpr index_t kKUnit = 16;
    /// K units a wave loads before it issues their WMMAs.
    static constexpr index_t kUnroll   = kUnroll_;
    static constexpr index_t kSlotRows = kMaxM;

    static_assert(kThreads % kWaveSize == 0 && kWaves >= 1, "wave32 workgroup");
    static_assert(kMaxM % 16 == 0 && kMaxM >= 16, "kMaxM must be a multiple of 16");
    static_assert(kNTiles >= 1 && kUnroll >= 1, "bad tiling");

    using barrier_pipeline = void;
    using Wmma             = WarpGemmAttributeWmmaImpl_f32_16x16x4_f32;

    struct Args
    {
        const bf16_t* h;   // [M, stride_h]
        const float* w;    // [num_experts, K]
        float* logits;     // [M, ld_logits]
        float* partial;    // [k_splits, n_tiles, M, kExpertsPerItem]; unused when k_splits == 1
        uint32_t* tickets; // [n_tiles], zeroed once; unused when k_splits == 1
        index_t M;
        index_t K;
        index_t stride_h;
        index_t ld_logits;
        index_t num_experts;
        index_t k_splits;
        index_t rank;
        index_t world;
        float scale; // output_multiplier
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // One 16-row tile of every wave's accumulators, plus the last-arriver flag.
        return kWaves * 16 * kExpertsPerItem * sizeof(float) + 16;
    }

    CK_TILE_HOST_DEVICE static index_t ExpertsPerRank(const Args& a)
    {
        return a.num_experts / a.world;
    }
    CK_TILE_HOST_DEVICE static index_t NumNTiles(const Args& a)
    {
        return ExpertsPerRank(a) / kExpertsPerItem;
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return NumNTiles(a) * a.k_splits; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t) { return NumWork(a); }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t)
    {
        return make_tuple(index_t{0}, a.M);
    }

    /// Floats of partial and words of tickets the caller allocates.
    CK_TILE_HOST static index_t PartialSize(const Args& a)
    {
        return a.k_splits > 1 ? a.k_splits * a.M * ExpertsPerRank(a) : 0;
    }
    CK_TILE_HOST static index_t NumTickets(const Args& a)
    {
        return a.k_splits > 1 ? NumNTiles(a) : 0;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
        return a.M >= 1 && a.M <= kMaxM && a.K % kKUnit == 0 && a.stride_h % 8 == 0 &&
               a.stride_h >= a.K && a.world >= 1 && a.rank >= 0 && a.rank < a.world &&
               a.num_experts % a.world == 0 && ExpertsPerRank(a) % kExpertsPerItem == 0 &&
               a.ld_logits >= a.num_experts && a.k_splits >= 1 && a.k_splits <= a.K / kKUnit &&
               aligned(a.h) && aligned(a.w) && a.logits != nullptr &&
               (a.k_splits == 1 || (a.partial != nullptr && a.tickets != nullptr));
    }

    CK_TILE_DEVICE static auto InitBarriers() { return 0; }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
        const index_t n_tiles = amd_wave_read_first_lane(NumNTiles(a));
        const index_t split   = amd_wave_read_first_lane(work / n_tiles);
        const index_t i_n     = amd_wave_read_first_lane(work - split * n_tiles);
        const index_t units   = a.K / kKUnit;
        const index_t u0      = amd_wave_read_first_lane(split * units / a.k_splits);
        const index_t u1      = amd_wave_read_first_lane((split + 1) * units / a.k_splits);
        const index_t m_tiles = amd_wave_read_first_lane(integer_divide_ceil(a.M, 16));
        const index_t n0 =
            amd_wave_read_first_lane(a.rank * ExpertsPerRank(a) + i_n * kExpertsPerItem);

        const index_t wave = get_warp_id();
        const index_t lane = get_lane_id();
        const index_t lr   = lane % 16; // A row / B column of this lane
        const index_t hf   = lane / 16; // which 8 of the unit's 16 k

        const float* w_lane  = a.w + static_cast<long_index_t>(n0 + lr) * a.K + 8 * hf;
        const bf16_t* h_lane = a.h + static_cast<long_index_t>(lr) * a.stride_h + 8 * hf;

        fp32x8_t acc[kMTiles][kNTiles];
        static_for<0, kMTiles, 1>{}([&](auto mt) {
            static_for<0, kNTiles, 1>{}([&](auto nt) { acc[mt][nt] = fp32x8_t{0.f}; });
        });

        // TODO: non-temporal / streaming loads for w (read once per step), and an LDS-shared h
        // tile if the per-wave h re-reads show up at M = 64.
        for(index_t base = u0 + wave; base < u1; base += kWaves * kUnroll)
        {
            fp32x4_t wv[kUnroll][kNTiles][2];
            uint32x4_t hv[kUnroll][kMTiles];
            static_for<0, kUnroll, 1>{}([&](auto u) {
                const index_t unit = base + u * kWaves;
                if(unit < u1)
                {
                    const index_t k = unit * kKUnit;
                    static_for<0, kNTiles, 1>{}([&](auto nt) {
                        const float* p = w_lane + static_cast<long_index_t>(nt * 16) * a.K + k;
                        wv[u][nt][0]   = impl::router_load16<fp32x4_t>(p);
                        wv[u][nt][1]   = impl::router_load16<fp32x4_t>(p + 4);
                    });
                    static_for<0, kMTiles, 1>{}([&](auto mt) {
                        hv[u][mt] = uint32x4_t{0u};
                        if(mt < m_tiles && mt * 16 + lr < a.M)
                        {
                            hv[u][mt] = impl::router_load16<uint32x4_t>(
                                h_lane + static_cast<long_index_t>(mt * 16) * a.stride_h + k);
                        }
                    });
                }
            });
            static_for<0, kUnroll, 1>{}([&](auto u) {
                if(base + u * kWaves < u1)
                {
                    static_for<0, kMTiles, 1>{}([&](auto mt) {
                        if(mt < m_tiles)
                        {
                            static_for<0, 4, 1>{}([&](auto i) {
                                // bf16 pair (2i, 2i + 1) is dword i: low half first.
                                const uint32_t d = hv[u][mt][i.value];
                                const fp32x2_t av{bit_cast<float>(d << 16),
                                                  bit_cast<float>(d & 0xffff0000u)};
                                static_for<0, kNTiles, 1>{}([&](auto nt) {
                                    const fp32x4_t& wq = wv[u][nt][i.value / 2];
                                    const fp32x2_t bv{wq[(i.value % 2) * 2],
                                                      wq[(i.value % 2) * 2 + 1]};
                                    Wmma{}(acc[mt][nt], av, bv);
                                });
                            });
                        }
                    });
                }
            });
        }

        // Add the waves' accumulators in wave order, one 16-row tile at a time.
        // C layout: element (m, n) is in lane n + 16 * (m / 8), vector element m % 8.
        float* red              = static_cast<float*>(smem);
        const bool direct       = a.k_splits == 1;
        float* const part       = direct ? nullptr
                                         : a.partial + static_cast<long_index_t>(split * n_tiles + i_n) *
                                                     a.M * kExpertsPerItem;
        constexpr index_t kTile = 16 * kExpertsPerItem;
        static_for<0, kMTiles, 1>{}([&](auto mt) {
            if(mt < m_tiles)
            {
                static_for<0, kNTiles, 1>{}([&](auto nt) {
                    static_for<0, 8, 1>{}([&](auto v) {
                        red[wave * kTile + (8 * hf + v) * kExpertsPerItem + nt * 16 + lr] =
                            acc[mt][nt][v.value];
                    });
                });
                block_sync_lds();
                for(index_t idx = threadIdx.x; idx < kTile; idx += kThreads)
                {
                    const index_t m = mt * 16 + idx / kExpertsPerItem;
                    const index_t e = idx % kExpertsPerItem;
                    float sum       = 0.f;
                    static_for<0, kWaves, 1>{}([&](auto w) { sum += red[w * kTile + idx]; });
                    if(m < a.M)
                    {
                        if(direct)
                            a.logits[m * a.ld_logits + n0 + e] = sum * a.scale;
                        else
                            part[m * kExpertsPerItem + e] = sum;
                    }
                }
                block_sync_lds();
            }
        });
        if(direct)
            return;

        // Last arriver of this n tile: publish the partial, take a ticket.
        uint32_t* flag = reinterpret_cast<uint32_t*>(red + kWaves * kTile);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            uint32_t* ticket = a.tickets + i_n;
            const uint32_t prev =
                __scoped_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
            const bool last = prev == static_cast<uint32_t>(a.k_splits - 1);
            if(last)
                __scoped_atomic_store_n(ticket, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            *flag = last ? 1u : 0u;
        }
        block_sync_lds();
        if(amd_wave_read_first_lane(*flag) == 0)
            return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        // TODO: this serial tail (k_splits * M * kExpertsPerItem floats, L2 resident) sits on the
        // critical path of the last item of every n tile; vectorise or spread it if it shows.
        const float* p0 = a.partial + static_cast<long_index_t>(i_n) * a.M * kExpertsPerItem;
        const long_index_t s_stride = static_cast<long_index_t>(n_tiles) * a.M * kExpertsPerItem;
        for(index_t idx = threadIdx.x; idx < a.M * kExpertsPerItem; idx += kThreads)
        {
            float sum = 0.f;
            for(index_t s = 0; s < a.k_splits; ++s)
                sum += p0[s * s_stride + idx];
            const index_t m                    = idx / kExpertsPerItem;
            const index_t e                    = idx % kExpertsPerItem;
            a.logits[m * a.ld_logits + n0 + e] = sum * a.scale;
        }
    }
};

} // namespace ck_tile

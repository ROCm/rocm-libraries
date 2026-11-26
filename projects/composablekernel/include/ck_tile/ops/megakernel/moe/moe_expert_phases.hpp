// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/moe/mx_expert_gemv.hpp"
#include "ck_tile/ops/megakernel/moe/route_table.hpp"

namespace ck_tile {

/// @file
/// MoE expert phases at decode M (8-64 rows, top-k 8), MXFP4 experts, A8W4 (MX fp8 activations):
///   MoeGateUpPhase   z[route]  = mxfp8(silu(p[row] @ gate[e]^T) * (p[row] @ up[e]^T))
///   MoeDownPhase     y[route]  = bf16(z[route] @ down[e]^T)
///   MoeCombinePhase  acc[row]  = sum_slot w * InputScaledRMSNorm(y[route])
/// with route = row * top_k + slot and e = topk_ids[route] (see moe_route_table).
///
/// Tensor parallelism (rank of world): both GEMMs are split on N. A rank computes columns
/// [rank * N / world, +N / world) of z (gate/up) and of y (down) and writes them at those columns
/// of full-width z / y buffers; the all-gathers that fill in the other ranks' columns are not part
/// of these phases (TODO(item 2)). With world = 1 the phases chain on one GPU.
///
/// Work items: (a, tile), a-major, where a indexes route.active[] and tile is an N tile of the
/// rank's slice. The number of active experts is only known on the device, so NumWork() is the
/// bound max_active * tiles and items with a >= num_active return at once. They still signal
/// their counter slot, which keeps every slot's per-launch count constant (megakernel counters
/// target epoch * ItemsPerSlot).
///
/// Counter slots are active-expert indices (kSlotRows = 1 "row" = one active expert), so the
/// down phase waits for exactly its expert's z with after_rows<gate/up stage>: its RowRange() of
/// item (a, tile) is [a, a + 1). The combine phase waits after_all on the down phase (one slot).
///
/// Within a work item: kTileN output features, kBlockSize_ / 32 waves = (kTileN / 16) row strips x
/// kKSplit K ranges. Each wave streams 16 weight rows x K / kKSplit, partial sums are reduced in
/// LDS, and the epilogue threads (one per token and 8 features) finish the tile. Tokens go in
/// groups of 16 (the WMMA N); an expert with more than 16 routes streams its weights once per
/// group (rare at M <= 64).

namespace moe_mx {

/// Shared tile logic of the two expert GEMM phases.
template <typename ActType_,
          index_t kNumMats_,
          index_t kK_,
          index_t kTileN_,
          index_t kKSplit_,
          index_t kBlockSize_>
struct expert_gemv_tile
{
    static constexpr index_t kK         = kK_;
    using ActType                       = ActType_;
    static constexpr index_t kNumMats   = kNumMats_;
    static constexpr index_t kTileN     = kTileN_;
    static constexpr index_t kKSplit    = kKSplit_;
    static constexpr index_t kThreads   = kBlockSize_; // threads on the device (wave32)
    static constexpr index_t kWaves     = kThreads / 32;
    /// Megakernel contract: waves * get_warp_size() of the compile pass (the host pass sees
    /// 64-lane waves and launches kBlockSize / get_warp_size() waves). Device code uses kThreads.
    static constexpr index_t kBlockSize = kWaves * get_warp_size();
    static constexpr index_t kStrips    = kTileN / kWaveRows;

    static_assert(kTileN % kScaleBlock == 0, "an N tile must hold whole requant blocks");
    static_assert(kStrips * kKSplit == kWaves, "waves = row strips x K splits");
    static_assert(2 * kTileN <= kThreads, "one epilogue thread per token and 8 features");
    static_assert(kK % (kKSplit * kWindowK) == 0, "whole windows per wave");
    static constexpr index_t kKPerWave = kK / kKSplit;

    // Partial sums [kKSplit][kNumMats][kTileN][kTokStride], padded against bank conflicts.
    static constexpr index_t kTokStride = kWaveTokens + 1;
    static constexpr index_t kRedFloats = kKSplit * kNumMats * kTileN * kTokStride;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kRedFloats * sizeof(float);
    }

    using Wave = mx_gemv_wave<ActType, kNumMats>;

    /// Streams the tile's weights against token group [g0, g0 + 16) and leaves the partial sums
    /// in LDS. @p x_row_of maps a token index (< cnt) to its activation row.
    template <typename XRowOf>
    CK_TILE_DEVICE static void Accumulate(const uint8_t* const (&w)[kNumMats],
                                          const uint8_t* const (&ws)[kNumMats],
                                          index_t w_row0,
                                          const uint8_t* x,
                                          const uint8_t* xs,
                                          index_t g0,
                                          index_t cnt,
                                          const XRowOf& x_row_of,
                                          float* red)
    {
        const index_t wave  = amd_wave_read_first_lane(static_cast<index_t>(threadIdx.x) / 32);
        const index_t lane  = static_cast<index_t>(__lane_id());
        const index_t strip = wave % kStrips;
        const index_t q     = wave / kStrips;
        constexpr index_t K = kK;

        const index_t row = w_row0 + strip * kWaveRows + lane % kWaveRows;
        const uint8_t* w_row[kNumMats];
        const uint8_t* ws_row[kNumMats];
        static_for<0, kNumMats, 1>{}([&](auto m) {
            w_row[m]  = w[m] + static_cast<long_index_t>(row) * (K / 2);
            ws_row[m] = ws[m] + static_cast<long_index_t>(row) * (K / kScaleBlock);
        });

        const index_t tok   = g0 + lane % kWaveTokens;
        const bool x_valid  = tok < cnt;
        const index_t x_row = x_row_of(x_valid ? tok : g0);

        fp32x8_t acc[kNumMats];
        static_for<0, kNumMats, 1>{}([&](auto m) { acc[m] = fp32x8_t{0}; });
        Wave::template Run<kKPerWave>(w_row,
                                      ws_row,
                                      x + static_cast<long_index_t>(x_row) * K,
                                      xs + static_cast<long_index_t>(x_row) * (K / kScaleBlock),
                                      x_valid,
                                      q * kKPerWave,
                                      acc);

        // C lane (token t, half h) element v is feature strip * 16 + 8 h + v.
        const index_t t = lane % kWaveTokens;
        const index_t h = lane / kWaveTokens;
        static_for<0, kNumMats, 1>{}([&](auto m) {
            static_for<0, 8, 1>{}([&](auto v) {
                const index_t f = strip * kWaveRows + 8 * h + v;
                red[((q * kNumMats + m) * kTileN + f) * kTokStride + t] = acc[m][v.value];
            });
        });
        __syncthreads();
    }

    /// Epilogue thread mapping: token = tid / (kTileN / 8), 8 features from 8 * (tid % (kTileN /
    /// 8)). Threads of one 32-feature block are 4 consecutive lanes.
    CK_TILE_DEVICE static bool EpilogueThread(index_t& tok, index_t& f)
    {
        const index_t tid = static_cast<index_t>(threadIdx.x);
        tok               = tid / (kTileN / 8);
        f                 = 8 * (tid % (kTileN / 8));
        return tid < 2 * kTileN;
    }

    /// Sum over K splits of matrix @p m, 8 features from @p f of token @p tok.
    CK_TILE_DEVICE static void
    Reduce(const float* red, index_t m, index_t f, index_t tok, float (&v)[8])
    {
        static_for<0, 8, 1>{}([&](auto i) {
            float s = 0;
            static_for<0, kKSplit, 1>{}([&](auto q) {
                s += red[((q * kNumMats + m) * kTileN + f + i) * kTokStride + tok];
            });
            v[i] = s;
        });
    }
};

/// Upper bound on route.num_active: an expert needs a route and there are M * top_k routes.
CK_TILE_HOST_DEVICE constexpr index_t max_active_experts(const moe_route_table& r)
{
    return min(r.num_experts, r.M * r.top_k);
}

} // namespace moe_mx

/// Gate/up projection + SwiGLU + MX requant of z (the down projection's activation).
///
/// @tparam ActType_ activation element type of p and z (fp8_t: A8W4).
/// @tparam kK_      reduction dim, the model dim (4096)
template <typename ActType_   = fp8_t,
          index_t kK_         = 4096,
          index_t kTileN_     = 32,
          index_t kKSplit_    = 4,
          index_t kBlockSize_ = 256>
struct MoeGateUpPhase
{
    using Tile    = moe_mx::expert_gemv_tile<ActType_, 2, kK_, kTileN_, kKSplit_, kBlockSize_>;
    using ActType = typename Tile::ActType;

    struct Args
    {
        moe_route_table route;
        const ActType* p;          ///< [M, K], one row per token
        const uint8_t* p_scale;    ///< [M, K / 32] e8m0
        const pk_fp4_t* gate;      ///< [E][w_rows][K / 2], expert stride w_rows * K / 2 bytes
        const uint8_t* gate_scale; ///< [E][w_rows][K / 32] e8m0
        const pk_fp4_t* up;
        const uint8_t* up_scale;
        ActType* z;       ///< [M * top_k, N], indexed by route
        uint8_t* z_scale; ///< [M * top_k, N / 32] e8m0
        index_t N;        ///< expert intermediate dim, full width (4096)
        index_t w_rows;   ///< weight rows stored per expert: N (full) or N / world (rank-local)
        index_t w_row0;   ///< first weight row of this rank's slice: rank * N / world or 0
        index_t rank;
        index_t world;
    };

    static constexpr index_t kBlockSize = Tile::kBlockSize;
    static constexpr index_t kSlotRows  = 1;
    using barrier_pipeline              = void;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Tile::GetSmemSize(); }

    CK_TILE_HOST_DEVICE static index_t Tiles(const Args& a) { return a.N / a.world / Tile::kTileN; }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a)
    {
        return moe_mx::max_active_experts(a.route) * Tiles(a);
    }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a)
    {
        return moe_mx::max_active_experts(a.route);
    }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t) { return Tiles(a); }
    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work) { return work / Tiles(a); }
    /// Rows are active-expert indices (see the file comment).
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t s = SlotOf(a, work);
        return make_tuple(s, s + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const index_t slice = a.world > 0 ? a.N / a.world : 0;
        return a.world > 0 && a.rank >= 0 && a.rank < a.world && a.N % a.world == 0 &&
               slice % Tile::kTileN == 0 && a.w_row0 + slice <= a.w_rows &&
               a.route.max_m >= a.route.M && a.route.top_k > 0;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
        constexpr index_t K = Tile::kK;
        const index_t tiles = Tiles(a);
        const index_t act   = amd_wave_read_first_lane(work / tiles);
        if(act >= amd_wave_read_first_lane(a.route.num_active[0]))
            return;
        const index_t tile   = work - act * tiles;
        const index_t e      = amd_wave_read_first_lane(a.route.active[act]);
        const index_t cnt    = amd_wave_read_first_lane(a.route.expert_count[e]);
        const int32_t* route = a.route.expert_routes + static_cast<long_index_t>(e) * a.route.max_m;
        const index_t top_k  = a.route.top_k;

        const index_t slice      = a.N / a.world;
        const index_t n_tile     = tile * Tile::kTileN; // within the rank's slice
        const long_index_t w_off = static_cast<long_index_t>(e) * a.w_rows * (K / 2);
        const long_index_t s_off =
            static_cast<long_index_t>(e) * a.w_rows * (K / moe_mx::kScaleBlock);
        const uint8_t* const w[2]  = {reinterpret_cast<const uint8_t*>(a.gate) + w_off,
                                      reinterpret_cast<const uint8_t*>(a.up) + w_off};
        const uint8_t* const ws[2] = {a.gate_scale + s_off, a.up_scale + s_off};

        float* red = static_cast<float*>(smem);
        for(index_t g0 = 0; g0 < cnt; g0 += moe_mx::kWaveTokens)
        {
            if(g0 > 0)
                __syncthreads(); // the previous group's epilogue has read red
            Tile::Accumulate(
                w,
                ws,
                a.w_row0 + n_tile,
                reinterpret_cast<const uint8_t*>(a.p),
                a.p_scale,
                g0,
                cnt,
                [&](index_t t) { return route[t] / top_k; },
                red);

            index_t tok, f;
            if(Tile::EpilogueThread(tok, f))
            {
                float g[8], u[8], z[8];
                Tile::Reduce(red, 0, f, tok, g);
                Tile::Reduce(red, 1, f, tok, u);
                float amax = 0;
                static_for<0, 8, 1>{}([&](auto i) {
                    z[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
                    amax = max(amax, abs(z[i]));
                });
                // The 4 lanes of one 32-feature block.
                amax                 = max(amax, __shfl_xor(amax, 1));
                amax                 = max(amax, __shfl_xor(amax, 2));
                const uint32_t sbits = moe_mx::mx_scale_bits<ActType>(amax);
                const float inv      = bit_cast<float>((254u << 23) - sbits);

                uint32_t packed[2];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
                static_for<0, 2, 1>{}([&](auto h) {
                    int old;
                    if constexpr(std::is_same_v<ActType, fp8_t>)
                    {
                        old = __builtin_amdgcn_cvt_pk_fp8_f32(
                            z[4 * h + 0] * inv, z[4 * h + 1] * inv, old, false);
                        old = __builtin_amdgcn_cvt_pk_fp8_f32(
                            z[4 * h + 2] * inv, z[4 * h + 3] * inv, old, true);
                    }
                    else
                    {
                        old = __builtin_amdgcn_cvt_pk_bf8_f32(
                            z[4 * h + 0] * inv, z[4 * h + 1] * inv, old, false);
                        old = __builtin_amdgcn_cvt_pk_bf8_f32(
                            z[4 * h + 2] * inv, z[4 * h + 3] * inv, old, true);
                    }
                    packed[h] = static_cast<uint32_t>(old);
                });
#pragma clang diagnostic pop

                const index_t t = g0 + tok;
                if(t < cnt)
                {
                    const long_index_t r = route[t];
                    const index_t col    = a.rank * slice + n_tile + f;
                    uint8_t* zp          = reinterpret_cast<uint8_t*>(a.z) + r * a.N + col;
                    *reinterpret_cast<uint32x2_t*>(zp) = uint32x2_t{packed[0], packed[1]};
                    if(f % moe_mx::kScaleBlock == 0)
                        a.z_scale[r * (a.N / moe_mx::kScaleBlock) + col / moe_mx::kScaleBlock] =
                            static_cast<uint8_t>(sbits >> 23);
                }
            }
        }
    }
};

/// Down projection: y[route, cols of this rank] = bf16(z[route] @ down[e]^T), full K.
template <typename ActType_   = fp8_t,
          index_t kK_         = 4096,
          index_t kTileN_     = 32,
          index_t kKSplit_    = 4,
          index_t kBlockSize_ = 256>
struct MoeDownPhase
{
    using Tile    = moe_mx::expert_gemv_tile<ActType_, 1, kK_, kTileN_, kKSplit_, kBlockSize_>;
    using ActType = typename Tile::ActType;

    struct Args
    {
        moe_route_table route;
        const ActType* z;          ///< [M * top_k, K] by route, all K gathered
        const uint8_t* z_scale;    ///< [M * top_k, K / 32]
        const pk_fp4_t* down;      ///< [E][w_rows][K / 2]
        const uint8_t* down_scale; ///< [E][w_rows][K / 32]
        bf16_t* y;                 ///< [M * top_k, N] by route
        index_t N;                 ///< model dim, full width (4096)
        index_t w_rows;
        index_t w_row0;
        index_t rank;
        index_t world;
    };

    static constexpr index_t kBlockSize = Tile::kBlockSize;
    static constexpr index_t kSlotRows  = 1;
    using barrier_pipeline              = void;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Tile::GetSmemSize(); }

    CK_TILE_HOST_DEVICE static index_t Tiles(const Args& a) { return a.N / a.world / Tile::kTileN; }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a)
    {
        return moe_mx::max_active_experts(a.route) * Tiles(a);
    }
    // One slot: the combine needs every expert of its row anyway.
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t) { return NumWork(a); }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }
    /// The z of active expert a: slot a of the gate/up phase.
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t s = work / Tiles(a);
        return make_tuple(s, s + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const index_t slice = a.world > 0 ? a.N / a.world : 0;
        return a.world > 0 && a.rank >= 0 && a.rank < a.world && a.N % a.world == 0 &&
               slice % Tile::kTileN == 0 && a.w_row0 + slice <= a.w_rows &&
               a.route.max_m >= a.route.M;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
        constexpr index_t K = Tile::kK;
        const index_t tiles = Tiles(a);
        const index_t act   = amd_wave_read_first_lane(work / tiles);
        if(act >= amd_wave_read_first_lane(a.route.num_active[0]))
            return;
        const index_t tile   = work - act * tiles;
        const index_t e      = amd_wave_read_first_lane(a.route.active[act]);
        const index_t cnt    = amd_wave_read_first_lane(a.route.expert_count[e]);
        const int32_t* route = a.route.expert_routes + static_cast<long_index_t>(e) * a.route.max_m;

        const index_t slice        = a.N / a.world;
        const index_t n_tile       = tile * Tile::kTileN;
        const uint8_t* const w[1]  = {reinterpret_cast<const uint8_t*>(a.down) +
                                      static_cast<long_index_t>(e) * a.w_rows * (K / 2)};
        const uint8_t* const ws[1] = {a.down_scale + static_cast<long_index_t>(e) * a.w_rows *
                                                         (K / moe_mx::kScaleBlock)};

        float* red = static_cast<float*>(smem);
        for(index_t g0 = 0; g0 < cnt; g0 += moe_mx::kWaveTokens)
        {
            if(g0 > 0)
                __syncthreads();
            Tile::Accumulate(
                w,
                ws,
                a.w_row0 + n_tile,
                reinterpret_cast<const uint8_t*>(a.z),
                a.z_scale,
                g0,
                cnt,
                [&](index_t t) { return route[t]; },
                red);

            index_t tok, f;
            if(Tile::EpilogueThread(tok, f) && g0 + tok < cnt)
            {
                float v[8];
                Tile::Reduce(red, 0, f, tok, v);
                uint32_t packed[4];
                static_for<0, 4, 1>{}([&](auto i) {
                    packed[i] = moe_mx::bf16_rne_bits(v[2 * i]) |
                                (static_cast<uint32_t>(moe_mx::bf16_rne_bits(v[2 * i + 1])) << 16);
                });
                const long_index_t r = route[g0 + tok];
                bf16_t* yp           = a.y + r * a.N + a.rank * slice + n_tile + f;
                *static_cast<int32x4_t*>(static_cast<void*>(yp)) =
                    int32x4_t{static_cast<int32_t>(packed[0]),
                              static_cast<int32_t>(packed[1]),
                              static_cast<int32_t>(packed[2]),
                              static_cast<int32_t>(packed[3])};
            }
        }
    }
};

/// Post-expert InputScaledRMSNorm of each route's y and the routing-weighted sum over a row's
/// slots. One work item owns one row, so acc needs no zeroing and no atomics:
///   v = bf16(y * (1 + w_norm)); n = bf16(v * rsqrt(mean(v^2) + eps)); acc[row] = sum_slot w * n
/// Accumulates in fp32 and rounds once (the HF reference index_adds bf16 per expert).
/// Requires the full y rows (after the bf16 all-gather when world > 1).
///
/// @tparam OutType_ bf16_t or float
/// @tparam kN_     row width (4096); kN % (8 * kBlockSize_) == 0
template <typename OutType_   = bf16_t,
          index_t kN_         = 4096,
          index_t kMaxTopK_   = 8,
          index_t kBlockSize_ = 256>
struct MoeCombinePhase
{
    using OutType                       = OutType_;
    static constexpr index_t kN         = kN_;
    static constexpr index_t kMaxTopK   = kMaxTopK_;
    static constexpr index_t kThreads   = kBlockSize_; // threads on the device (wave32)
    static constexpr index_t kWaves     = kThreads / 32;
    /// Megakernel contract: waves * get_warp_size() of the compile pass. Device code uses
    /// kThreads.
    static constexpr index_t kBlockSize = kWaves * get_warp_size();
    static constexpr index_t kChunks    = kN / (8 * kThreads); // 8 bf16 per chunk
    static_assert(kN % (8 * kThreads) == 0);
    static_assert(std::is_same_v<OutType, bf16_t> || std::is_same_v<OutType, float>);

    struct Args
    {
        moe_route_table route;
        const bf16_t* y;      ///< [M * top_k, kN] by route
        const bf16_t* norm_w; ///< [kN] post_expert_norm weight
        OutType* out;         ///< [M, kN]
        float eps;            ///< post_norm_eps (1e-8)
    };

    static constexpr index_t kSlotRows = 1;
    using barrier_pipeline             = void;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kWaves * kMaxTopK * sizeof(float);
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.route.M; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.route.M; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t work) { return work; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t work)
    {
        return make_tuple(work, work + 1);
    }
    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.route.top_k > 0 && a.route.top_k <= kMaxTopK;
    }

    CK_TILE_DEVICE static void Load(const bf16_t* src, index_t c, float (&v)[8])
    {
        const int32x4_t raw = *static_cast<const int32x4_t*>(static_cast<const void*>(src + 8 * c));
        static_for<0, 4, 1>{}([&](auto i) {
            const uint32_t u = static_cast<uint32_t>(raw[i.value]);
            v[2 * i]         = bit_cast<float>(u << 16);
            v[2 * i + 1]     = bit_cast<float>(u & 0xFFFF0000u);
        });
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t row, void* smem, const Token&)
    {
        const index_t top_k = a.route.top_k;
        const index_t tid   = static_cast<index_t>(threadIdx.x);
        float* red          = static_cast<float*>(smem);

        float w1[kChunks][8]; // 1 + norm weight
        static_for<0, kChunks, 1>{}([&](auto c) {
            Load(a.norm_w, tid + c * kThreads, w1[c]);
            static_for<0, 8, 1>{}([&](auto i) { w1[c][i] += 1.0f; });
        });

        // Pass 1: sum of squares of every slot, one block reduction for all of them.
        float ss[kMaxTopK];
        static_for<0, kMaxTopK, 1>{}([&](auto s) {
            ss[s] = 0;
            if(s < top_k)
            {
                const long_index_t r = static_cast<long_index_t>(row) * top_k + s;
                static_for<0, kChunks, 1>{}([&](auto c) {
                    float v[8];
                    Load(a.y + r * kN, tid + c * kThreads, v);
                    static_for<0, 8, 1>{}([&](auto i) {
                        const float x = moe_mx::bf16_round(v[i] * w1[c][i]);
                        ss[s] += x * x;
                    });
                });
            }
        });
        static_for<0, kMaxTopK, 1>{}([&](auto s) {
            for(index_t o = 16; o > 0; o /= 2)
                ss[s] += __shfl_xor(ss[s], o);
        });
        const index_t wave = tid / 32;
        if(tid % 32 == 0)
            static_for<0, kMaxTopK, 1>{}([&](auto s) { red[wave * kMaxTopK + s] = ss[s]; });
        __syncthreads();

        float scale[kMaxTopK];
        static_for<0, kMaxTopK, 1>{}([&](auto s) {
            float t = 0;
            static_for<0, kWaves, 1>{}([&](auto wv) { t += red[wv * kMaxTopK + s]; });
            scale[s] = rsqrtf(t / kN + a.eps);
        });

        // Pass 2: the y rows were just read by this workgroup, so this reread hits the caches.
        float acc[kChunks][8] = {};
        static_for<0, kMaxTopK, 1>{}([&](auto s) {
            const long_index_t r = static_cast<long_index_t>(row) * top_k + s;
            if(s < top_k && a.route.topk_ids[r] >= 0 && a.route.topk_ids[r] < a.route.num_experts)
            {
                const float wt = a.route.topk_weights[r];
                static_for<0, kChunks, 1>{}([&](auto c) {
                    float v[8];
                    Load(a.y + r * kN, tid + c * kThreads, v);
                    static_for<0, 8, 1>{}([&](auto i) {
                        const float x = moe_mx::bf16_round(v[i] * w1[c][i]);
                        acc[c][i] += wt * moe_mx::bf16_round(x * scale[s]);
                    });
                });
            }
        });

        static_for<0, kChunks, 1>{}([&](auto c) {
            OutType* dst = a.out + static_cast<long_index_t>(row) * kN + 8 * (tid + c * kThreads);
            if constexpr(std::is_same_v<OutType, float>)
            {
                static_for<0, 8, 1>{}([&](auto i) { dst[i] = acc[c][i]; });
            }
            else
            {
                int32x4_t pk;
                static_for<0, 4, 1>{}([&](auto i) {
                    pk[i.value] = static_cast<int32_t>(
                        moe_mx::bf16_rne_bits(acc[c][2 * i]) |
                        (static_cast<uint32_t>(moe_mx::bf16_rne_bits(acc[c][2 * i + 1])) << 16));
                });
                *static_cast<int32x4_t*>(static_cast<void*>(dst)) = pk;
            }
        });
        // The next item's smem writes wait at the megakernel's per-item barrier.
    }
};

} // namespace ck_tile

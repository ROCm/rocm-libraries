// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A-stationary fused GEMM + permute for gfx1250: the Y[h,b,t,d] of gemm_permute.hpp, computed by a
// hand-written wave32 WMMA kernel.
// - One workgroup (8 waves) per (b, 256-row t tile). Its A tile [256, K] is copied into LDS once
//   (global_load_async_to_lds_b128, nontemporal) in three K thirds; the waves start computing once
//   the first third has landed. It uses all 320 KiB of LDS, so one workgroup runs per CU.
// - The waves form 2 row groups x 4 column groups. Each wave walks eight [128, 64] output tiles,
//   one per head: at step i the 4 column groups of a row group cover head i's D columns.
// - 16x16x32 WMMA on the transposed tile, C'[d][t] = W^T A^T: the weight fragments (WMMA A) are
//   loaded from global memory straight into registers, two K steps ahead (ring of 3); the A
//   fragments (WMMA B) come from LDS, two WMMA groups ahead; the last K steps of a tile prefetch
//   the first fragments of the next one.
// - Each finished tile is converted to BF16 into a per-wave LDS stage and drained during the next
//   tile's K loop by nontemporal global_store_async_from_lds_b128 (three per K step).
// - The weight is the original W, repacked on every call into the ck_tile::shuffle_b layout of the
//   head-major GEMM (repack_weight).
// Included by gemm_permute_astationary.hpp, which adds repack_weight and launch.
#pragma once

#include "gemm_permute.hpp"

namespace gemm_permute::astationary {

// Workgroup rows, wave tile columns, waves; the waves form RowGroups x ColGroups.
inline constexpr int BM = 256, WN = 64, Waves = 8, Threads = Waves * 32;
inline constexpr int RowGroups = 2, ColGroups = Waves / RowGroups, RowsW = BM / RowGroups;
// 16x16x32 WMMA: K steps, 16-row and 16-column blocks of a wave tile, WMMA groups (one 16-row
// block of one K step) per tile, wave tiles per wave.
inline constexpr int KS = K / 32, MT = RowsW / 16, NT = WN / 16, Groups = KS * MT;
inline constexpr int WaveTiles = N / WN / ColGroups;
// Weight fragment ring (loads run BRing - 1 K steps ahead); A fragments run APF groups ahead in a
// ring of APF + 2 (a load overwrites the slot of group g - 2, not g - 1).
inline constexpr int BRing = 3, APF = 2, ARing = APF + 2;
// WMMA group of each K step that issues the step's weight loads and its share of the Y stores.
inline constexpr int IssueGroup = 1;
// A in LDS: unpadded [BM, K] rows, 16 B unit u of row r at u ^ (r % 16); copied in AParts K pieces.
inline constexpr int AUnits = BM * K / 8, AChunks = AUnits / Threads, AParts = 3;
inline constexpr int PerPart = AChunks / AParts, PartUnits = K / 8 / AParts;
// Y stage per wave: the [RowsW, WN] BF16 tile, 16 B unit u of row r at u ^ (r % 8). A store
// instruction writes RowsPerStore rows of WN * 2 bytes; StoresPerK of them run per K step.
inline constexpr int StUnits = RowsW * WN / 8, RowUnits = WN / 8, RowsPerStore = 32 / RowUnits;
inline constexpr int Stores = RowsW / RowsPerStore, StoresPerK = (Stores + KS - 1) / KS;
inline constexpr int StoreNT = 1; // cache policy of the async Y stores: nontemporal
static_assert(K % 32 == 0 && D % WN == 0 && N / WN % ColGroups == 0);
static_assert(AUnits % Threads == 0 && KS % AParts == 0 && PartUnits % 16 == 0);
static_assert(KS % BRing == 0 && Groups % ARing == 0); // the cross-tile prefetch keeps ring slots
static_assert(StoresPerK * KS >= Stores && RowUnits % RowsPerStore == 0);
static_assert((AUnits + Waves * StUnits) * 16 == 327680);

// Weight layout (ck_tile::shuffle_b, 16x16x32): flat row R holds GEMM columns R * WT + j, and its
// 32-wide K step s is the 1 KiB chunk Wf[(R * K / 32 + s) * 512 ..]; lane l reads its WMMA A
// fragment from the chunk's 16 B units l and l + 32.
inline constexpr int WT = 16;
__host__ __device__ constexpr int flat_column(int R, int j) { return R * WT + j; }

union Frag
{
    ck_tile::llvm_bf16x16_t v;
    uint4 q[2];
};

__global__ void __launch_bounds__(Threads, 1) kernel(const Bf16* __restrict__ a,
                                                     const Bf16* __restrict__ wf,
                                                     Bf16* __restrict__ y,
                                                     int time,
                                                     int batch)
{
#if defined(__gfx1250__)
    using namespace ck_tile;
    __shared__ uint4 lds[AUnits + Waves * StUnits];
    const int time_tiles = (time + BM - 1) / BM;
    const int b = blockIdx.x / time_tiles, t0 = blockIdx.x % time_tiles * BM;
    const int rows = ck_tile::min(BM, time - t0);
    const int lane = threadIdx.x % 32;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    const int cgw = wave % ColGroups, rw0 = wave / ColGroups * RowsW; // column group, first row

    // A tile -> LDS. Piece p holds the units [p * PartUnits, (p + 1) * PartUnits) of every row,
    // consecutive lanes walk a row piece. A tail tile zero-fills the rows past `rows`.
    const Bf16* a_tile = a + (int64_t{t0} * batch + b) * K;
    auto load_a_tile   = [&](auto guarded) {
#pragma unroll
        for(int j = 0; j < AChunks; ++j)
        {
            const int e = threadIdx.x + j % PerPart * Threads;
            const int r = e / PartUnits, k8 = j / PerPart * PartUnits + e % PartUnits;
            uint4* dst = &lds[r * (K / 8) + (k8 ^ r % 16)];
            if(!guarded || r < rows)
                asm volatile("global_load_async_to_lds_b128 %0, %1, %2 th:TH_LOAD_NT" ::"v"(
                                 static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(dst)))),
                             "v"(static_cast<uint32_t>((r * batch * K + k8 * 8) * sizeof(Bf16))),
                             "s"(a_tile)
                             : "memory");
            else
                *dst = uint4{0, 0, 0, 0};
        }
    };
    if(rows == BM)
        load_a_tile(bool_constant<false>{});
    else
        load_a_tile(bool_constant<true>{});
    // Piece p is in this lane's LDS once at most (AParts - 1 - p) * PerPart copies are pending. A
    // tail tile waits for all of A here (the later waits are then no-ops). The asynccnt waits are
    // raw: ck_tile::s_wait_asynccnt waits on asyncmarks where the compiler has them, and these
    // copies are not marked.
    auto wait_part = [&](auto p) {
        asm volatile("s_wait_asynccnt %0" ::"n"((AParts - 1 - decltype(p)::value) * PerPart)
                     : "memory");
        __syncthreads();
    };
    if(rows < BM)
        wait_part(number<AParts - 1>{});
    else
        wait_part(number<0>{});

    // A^T fragment (WMMA B) of group g (16-row block g % MT, K step g / MT): lane holds
    // t = 16 * (g % MT) + lane % 16, k = 32 * (g / MT) + 8 * (lane / 16) + [0, 8) and + 16.
    auto load_a = [&](int g) {
        Frag f;
        const int r = rw0 + (g % MT) * 16 + lane % 16, c = g / MT * 4 + lane / 16;
        f.q[0] = lds[r * (K / 8) + (c ^ (lane % 16))];
        f.q[1] = lds[r * (K / 8) + ((c + 2) ^ (lane % 16))];
        return f;
    };
    // W^T fragment (WMMA A) of 16-column block nb, K step ks: a contiguous 512 B per half.
    auto load_w = [&](int nb, int ks) {
        Frag f;
        const uint4* p = reinterpret_cast<const uint4*>(wf) + (int64_t{nb} * KS + ks) * 64 + lane;
        f.q[0]         = p[0];
        f.q[1]         = p[32];
        return f;
    };

    // Deferred Y stores of the previous wave tile from the stage. The instruction offset is added
    // to both the global and the LDS address, so store j takes offset j * StJ and its global base
    // absorbs the difference; lanes then need one LDS address per j % Par.
    uint4* const stage   = lds + AUnits + wave * StUnits;
    const bool full_slab = t0 + rw0 + RowsW <= time;
    constexpr int StJ = RowsPerStore * WN * 2, GlJ = RowsPerStore * D * 2;
    constexpr int Par      = RowUnits / RowsPerStore;
    const unsigned st_base = static_cast<unsigned>(reinterpret_cast<uintptr_t>(stage));
    unsigned st_lane[Par];
#pragma unroll
    for(int p = 0; p < Par; ++p)
        st_lane[p] = st_base + (lane / RowUnits) * WN * 2 +
                     ((lane % RowUnits) ^ (p * RowsPerStore + lane / RowUnits) % RowUnits) * 16;
    const unsigned y_lane = (lane / RowUnits) * D * 2 + lane % RowUnits * 16;
    const char* y_slab    = nullptr; // Y rows of the staged tile (uniform)
    auto store_y          = [&](auto jc, bool guard) {
        constexpr int j = decltype(jc)::value;
        if constexpr(j < Stores)
            if(!guard || t0 + rw0 + j * RowsPerStore + lane / RowUnits < time)
                __builtin_amdgcn_global_store_async_from_lds_b128(
                    reinterpret_cast<__attribute__((address_space(1))) int32x4_t*>(
                        to_global(y_slab + j * (GlJ - StJ)) + y_lane),
                    reinterpret_cast<__attribute__((address_space(3))) int32x4_t*>(
                        uintptr_t{st_lane[j % Par]}),
                    j * StJ,
                    StoreNT);
    };

    // Software pipeline, pinned with sched_barrier so the compiler cannot sink loads to their uses.
    Frag wq[BRing][NT], aq[ARing];
    const int n0_first = cgw * WN;
#pragma unroll
    for(int s = 0; s < BRing - 1; ++s)
#pragma unroll
        for(int nb = 0; nb < NT; ++nb)
            wq[s][nb] = load_w(n0_first / 16 + nb, s);
#pragma unroll
    for(int g = 0; g < APF; ++g)
        aq[g] = load_a(g);
    for(int i = 0; i < WaveTiles; ++i)
    {
        const int n0 = (cgw + ColGroups * i) * WN, h = n0 / D, d0 = n0 % D; // head-major column
        const int n0_next = n0 + ColGroups * WN;
        ck_tile::fp32x8_t c[MT][NT];
#pragma unroll
        for(int tb = 0; tb < MT; ++tb)
#pragma unroll
            for(int nb = 0; nb < NT; ++nb)
                c[tb][nb] = ck_tile::fp32x8_t{};
        // ks must be a constant: it selects the async stores' immediate offsets.
        static_for<0, KS, 1>{}([&](auto ksc) __attribute__((always_inline)) {
            constexpr int ks = decltype(ksc)::value;
#pragma unroll
            for(int tb = 0; tb < MT; ++tb)
            {
                const int g = ks * MT + tb, kb = ks + BRing - 1;
                if(tb == IssueGroup)
                {
                    if(kb < KS)
#pragma unroll
                        for(int nb = 0; nb < NT; ++nb)
                            wq[kb % BRing][nb] = load_w(n0 / 16 + nb, kb);
                    else if(i + 1 < WaveTiles)
#pragma unroll
                        for(int nb = 0; nb < NT; ++nb)
                            wq[kb % BRing][nb] = load_w(n0_next / 16 + nb, kb - KS);
                }
                // Tile 0: the first A fragment of piece p waits for piece p.
                if(i == 0 && (g + APF) % (Groups / AParts) == 0 && g + APF < Groups)
                    static_for<1, AParts, 1>{}([&](auto p) {
                        if(p == (g + APF) / (Groups / AParts))
                            wait_part(p);
                    });
                if(g + APF < Groups)
                    aq[(g + APF) % ARing] = load_a(g + APF);
                else if(i + 1 < WaveTiles)
                    aq[(g + APF) % ARing] = load_a(g + APF - Groups);
                if(tb == IssueGroup && i > 0)
                {
                    // The stage writes of the previous tile must have landed in LDS.
                    if(ks == 0)
                        s_wait_dscnt_pinned<0>();
                    if(full_slab)
                    {
                        static_for<0, StoresPerK, 1>{}([&](auto q) {
                            store_y(number<ks * StoresPerK + decltype(q)::value>{}, false);
                        });
                        // keeps the unguarded stores out of an exec-masked merge block
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    else
                        static_for<0, StoresPerK, 1>{}([&](auto q) {
                            store_y(number<ks * StoresPerK + decltype(q)::value>{}, true);
                        });
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int nb = 0; nb < NT; ++nb)
                    c[tb][nb] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false,
                                                                        wq[ks % BRing][nb].v,
                                                                        false,
                                                                        aq[g % ARing].v,
                                                                        0,
                                                                        c[tb][nb],
                                                                        false,
                                                                        false);
                __builtin_amdgcn_sched_barrier(0);
            }
        });
        // Lane holds t = 16 * tb + lane % 16, d = 16 * nb + 8 * (lane / 16) + [0, 8). The async
        // stores of the previous tile must have read the stage before it is overwritten.
        asm volatile("s_wait_asynccnt 0x0" ::: "memory");
#pragma unroll
        for(int tb = 0; tb < MT; ++tb)
#pragma unroll
            for(int nb = 0; nb < NT; ++nb)
            {
                union
                {
                    Bf16 e[8];
                    uint4 q;
                } o;
#pragma unroll
                for(int e = 0; e < 8; ++e)
                    o.e[e] = type_convert<Bf16>(c[tb][nb][e]);
                const int r                                                 = tb * 16 + lane % 16;
                stage[r * RowUnits + ((nb * 2 + lane / 16) ^ r % RowUnits)] = o.q;
            }
        y_slab = reinterpret_cast<const char*>(
            y + ((int64_t{h} * batch + b) * time + t0 + rw0) * D + d0);
    }
    s_wait_dscnt_pinned<0>();
    static_for<0, Stores, 1>{}([&](auto j) { store_y(j, !full_slab); });
    // The stage must stay allocated until the async stores have read it.
    asm volatile("s_wait_asynccnt 0x0" ::: "memory");
#else
    static_cast<void>(a);
    static_cast<void>(wf);
    static_cast<void>(y);
    static_cast<void>(time);
    static_cast<void>(batch);
#endif
}

} // namespace gemm_permute::astationary

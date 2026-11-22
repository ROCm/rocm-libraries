// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_a_stationary_base.hpp"

namespace ck_tile {

/// @brief A-stationary mainloop for gfx1250 (wave32 16x16x32 WMMA, async LDS transfers).
/// - The workgroup's A tile [MPerBlock, K] is copied into LDS once (global_load_async_to_lds_b128,
///   nontemporal) in AParts K pieces; the waves start once the first piece has landed.
/// - Each wave walks its [RowsPerWave, NPerWave] output tiles over N with no workgroup barrier.
///   The WMMAs compute the transposed tile, C^T = W^T A^T: weight fragments (WMMA A) come from
///   the flat weight straight into registers, BRing - 1 K steps ahead; A fragments (WMMA B) come
///   from LDS, APF groups ahead; the last K steps of a tile prefetch the next tile's first ones.
/// - A finished tile is converted to E in a per-wave LDS stage and drained during the next tile's
///   K loop by nontemporal global_store_async_from_lds_b128, StoresPerK per K step.
/// Flat weight: flat row R holds GEMM columns 16 R .. 16 R + 15; its 32-wide K step s is the 1 KiB
/// chunk at element (R * K / 32 + s) * 512 (the ck_tile::shuffle_b layout of a 16x16x32 GEMM on
/// gfx12); FlatWeightOffset gives the position of each weight element.
/// Compiles to nothing on other targets, whose devices IsSupportedArgument rejects.
template <typename Problem_>
struct GemmPipelineAStationaryAsync : public GemmPipelineAStationaryBase<Problem_>
{
    using Base = GemmPipelineAStationaryBase<Problem_>;
    using typename Base::ADataType;
    using typename Base::EDataType;
    using BDataType = typename Base::Problem::BDataType;
    using Base::kColGroups;
    using Base::kK;
    using Base::kMPerBlock;
    using Base::kNPerWave;
    using Base::kRowsPerWave;
    using Base::kStrideM0;
    using Base::kWaves;

    static_assert(std::is_same_v<ADataType, bf16_t> && std::is_same_v<BDataType, bf16_t> &&
                      std::is_same_v<typename Base::Problem::AccDataType, float> &&
                      sizeof(EDataType) == 2,
                  "BF16 A and B, FP32 accumulation, 16-bit E");

    static constexpr index_t BlockSize = kWaves * 32;
    static constexpr int kBlockPerCu   = 1;

    // 16x16x32 WMMA: K steps, 16-row and 16-column blocks of a wave tile, WMMA groups (one 16-row
    // block of one K step) per tile.
    static constexpr index_t KS = kK / 32, MT = kRowsPerWave / 16, NT = kNPerWave / 16;
    static constexpr index_t Groups = KS * MT;
    // Weight fragment ring (loads run BRing - 1 K steps ahead); A fragments run APF groups ahead in
    // a ring of APF + 2 (a load overwrites the slot of group g - 2, not g - 1).
    static constexpr index_t BRing = KS % 3 == 0 ? 3 : 2, APF = 2, ARing = APF + 2;
    // WMMA group of each K step that issues the step's weight loads and its share of the E stores.
    static constexpr index_t IssueGroup = 1;
    // A in LDS: unpadded [MPerBlock, K] rows, 16 B unit u of row r at u ^ (r % 16); AParts pieces.
    static constexpr index_t AParts = kK % 384 == 0 ? 3 : 1;
    static constexpr index_t AUnits = kMPerBlock * kK / 8, AChunks = AUnits / BlockSize;
    static constexpr index_t PerPart = AChunks / AParts, PartUnits = kK / 8 / AParts;
    // E stage per wave: the [RowsPerWave, NPerWave] tile, 16 B unit u of row r at u ^ (r % 8). A
    // store instruction writes RowsPerStore rows of NPerWave elements; StoresPerK run per K step.
    static constexpr index_t StUnits = kRowsPerWave * kNPerWave / 8, RowUnits = kNPerWave / 8;
    static constexpr index_t RowsPerStore = 32 / RowUnits, Stores = kRowsPerWave / RowsPerStore;
    static constexpr index_t StoresPerK = (Stores + KS - 1) / KS;

    static_assert(kK % 128 == 0 && kRowsPerWave % 16 == 0 && kNPerWave % 16 == 0);
    static_assert(AUnits % BlockSize == 0 && AChunks % AParts == 0 && KS % AParts == 0 &&
                  PartUnits % 16 == 0);
    static_assert(KS % BRing == 0 && Groups % ARing == 0); // the cross-tile prefetch keeps slots
    static_assert(StoresPerK * KS >= Stores && RowUnits % RowsPerStore == 0);
    // The weight loads and E stores issue in WMMA group IssueGroup of a K step; one store
    // instruction's 32 lanes cover RowsPerStore whole rows (RowUnits a power of two: XOR swizzle).
    static_assert(MT > IssueGroup && RowUnits * RowsPerStore == 32);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return (AUnits + kWaves * StUnits) * 16;
    }
    static_assert(GetSmemSize() <= 327680, "A tile plus the wave stages exceed the gfx1250 LDS");

    /// Element offset of the weight W[k, n] in the flat weight: lane l of a K step reads the
    /// 16 B units l and l + 32 of its chunk, the WMMA A fragment of column 16 R + l % 16.
    CK_TILE_HOST_DEVICE static constexpr long_index_t FlatWeightOffset(index_t k, index_t n)
    {
        return (long_index_t{n / 16} * KS + k / 32) * 512 + (k % 32 / 8 * 16 + n % 16) * 8 + k % 8;
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AStationaryAsync",
                      concat('x', kMPerBlock, kK, kNPerWave), BlockSize,
                      concat('x', Base::kRowGroups, kColGroups),
                      kStrideM0);
        // clang-format on
    }

    CK_TILE_HOST static bool IsSupportedArgument(const GemmAStationaryKernelArgs& kargs)
    {
        // The mainloop compiles to nothing on other targets, so such a device is rejected here.
        if(!is_gfx125_supported())
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("GemmAStationaryAsync requires a gfx1250 device");
            }
            return false;
        }
        return Base::IsSupportedArgument(kargs);
    }

    /// Workgroup (m0_0, m1): rows m0_0 .. m0_0 + rows - 1 of m0 at m1, all N columns.
    CK_TILE_DEVICE void operator()(const GemmAStationaryKernelArgs& kargs,
                                   index_t m0_0,
                                   index_t m1,
                                   index_t rows,
                                   void* p_smem) const
    {
#if defined(__gfx125__)
        uint4* const lds = static_cast<uint4*>(p_smem);
        const int lane   = threadIdx.x % 32;
        const int wave   = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
        const int cgw = wave % kColGroups, rw0 = wave / kColGroups * kRowsPerWave; // column group

        // A tile -> LDS. Piece p holds the units [p * PartUnits, (p + 1) * PartUnits) of every
        // row, consecutive lanes walk a row piece. A tail tile zero-fills the rows past `rows`.
        const ADataType* a_tile = Base::ARow0(kargs, m0_0, m1);
        const index_t a_row     = kargs.e.M1 * kargs.stride_A; // elements between tile rows
        auto load_a_tile        = [&](auto guarded) {
#pragma unroll
            for(int j = 0; j < AChunks; ++j)
            {
                const int e = threadIdx.x + j % PerPart * BlockSize;
                const int r = e / PartUnits, k8 = j / PerPart * PartUnits + e % PartUnits;
                amd_async_global_load_to_lds<ADataType, 8, 0, true, amd_buffer_coherence_enum::NT>(
                    reinterpret_cast<ADataType*>(&lds[r * (kK / 8) + (k8 ^ r % 16)]),
                    a_tile,
                    r * a_row + k8 * 8,
                    !guarded || r < rows);
            }
        };
        if(rows == kMPerBlock)
            load_a_tile(bool_constant<false>{});
        else
            load_a_tile(bool_constant<true>{});
        // Piece p is in this lane's LDS once at most (AParts - 1 - p) * PerPart copies are
        // pending. A tail tile waits for all of A here (the later waits are then no-ops).
        auto wait_part = [&](auto p) {
            s_wait_asynccnt_pinned<(AParts - 1 - decltype(p)::value) * PerPart>();
            __syncthreads();
        };
        if(rows < kMPerBlock)
            wait_part(number<AParts - 1>{});
        else
            wait_part(number<0>{});

        union Frag
        {
            llvm_bf16x16_t v;
            uint4 q[2];
        };
        // A^T fragment (WMMA B) of group g (16-row block g % MT, K step g / MT): lane holds
        // row 16 * (g % MT) + lane % 16, k = 32 * (g / MT) + 8 * (lane / 16) + [0, 8) and + 16.
        auto load_a = [&](int g) {
            Frag f;
            const int r = rw0 + (g % MT) * 16 + lane % 16, c = g / MT * 4 + lane / 16;
            f.q[0] = lds[r * (kK / 8) + (c ^ (lane % 16))];
            f.q[1] = lds[r * (kK / 8) + ((c + 2) ^ (lane % 16))];
            return f;
        };
        // W^T fragment (WMMA A) of 16-column block nb, K step ks: a contiguous 512 B per half.
        auto load_w = [&](int nb, int ks) {
            Frag f;
            const uint4* p = reinterpret_cast<const uint4*>(kargs.b_flat_ptr) +
                             (int64_t{nb} * KS + ks) * 64 + lane;
            f.q[0] = p[0];
            f.q[1] = p[32];
            return f;
        };

        // Deferred E stores of the previous wave tile from the stage. The instruction offset is
        // added to both the global and the LDS address, so store j takes offset j * StJ and its
        // global base absorbs the difference; lanes then need one LDS address per j % Par.
        uint4* const stage   = lds + AUnits + wave * StUnits;
        const bool full_slab = rw0 + kRowsPerWave <= rows;
        constexpr int StJ    = RowsPerStore * kNPerWave * 2;
        const int row_bytes =
            (kStrideM0 > 0 ? kStrideM0 : kargs.e.stride_M0) * index_t{sizeof(EDataType)};
        const int GlJ          = RowsPerStore * row_bytes;
        constexpr int Par      = RowUnits / RowsPerStore;
        const unsigned st_base = static_cast<unsigned>(reinterpret_cast<uintptr_t>(stage));
        unsigned st_lane[Par];
#pragma unroll
        for(int p = 0; p < Par; ++p)
            st_lane[p] = st_base + (lane / RowUnits) * kNPerWave * 2 +
                         ((lane % RowUnits) ^ (p * RowsPerStore + lane / RowUnits) % RowUnits) * 16;
        const unsigned e_lane = (lane / RowUnits) * row_bytes + lane % RowUnits * 16;
        char* e_slab          = nullptr; // E rows of the staged tile (uniform)
        auto e_addr           = [&](auto jc) -> void* {
            return e_slab + decltype(jc)::value * (GlJ - StJ) + e_lane;
        };
        auto issue_e = [&](auto jc, void* addr) {
            constexpr int j = decltype(jc)::value;
            amd_async_global_store_from_lds_b128<j * StJ, amd_buffer_coherence_enum::NT>(
                addr, st_lane[j % Par]);
        };
        auto store_e = [&](auto jc, bool guard) {
            constexpr int j = decltype(jc)::value;
            if constexpr(j < Stores)
                if(!guard || rw0 + j * RowsPerStore + lane / RowUnits < rows)
                    issue_e(jc, e_addr(jc));
        };

        // Software pipeline, pinned with sched_barrier so the compiler cannot sink loads to their
        // uses.
        const int wave_tiles = Base::WaveTiles(kargs.e.N0 * kargs.e.N1);
        Frag wq[BRing][NT], aq[ARing];
        const int n0_first = Base::TileN0(cgw, 0);
#pragma unroll
        for(int s = 0; s < BRing - 1; ++s)
#pragma unroll
            for(int nb = 0; nb < NT; ++nb)
                wq[s][nb] = load_w(n0_first / 16 + nb, s);
#pragma unroll
        for(int g = 0; g < APF; ++g)
            aq[g] = load_a(g);
        for(int i = 0; i < wave_tiles; ++i)
        {
            const int n0 = Base::TileN0(cgw, i), n0_next = n0 + kColGroups * kNPerWave;
            fp32x8_t c[MT][NT];
#pragma unroll
            for(int tb = 0; tb < MT; ++tb)
#pragma unroll
                for(int nb = 0; nb < NT; ++nb)
                    c[tb][nb] = fp32x8_t{};
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
                        else if(i + 1 < wave_tiles)
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
                    else if(i + 1 < wave_tiles)
                        aq[(g + APF) % ARing] = load_a(g + APF - Groups);
                    if(tb == IssueGroup && i > 0)
                    {
                        // The stage writes of the previous tile must have landed in LDS.
                        if(ks == 0)
                            s_wait_dscnt_pinned<0>();
                        if(full_slab)
                        {
                            // All addresses of the step first: with a run-time row stride
                            // the compiler otherwise computes each into the same register
                            // pair right after the previous store, which measured slower.
                            constexpr int j0 = ks * StoresPerK;
                            constexpr int nq = min(StoresPerK, max(Stores - j0, 0));
                            void* addr[StoresPerK];
                            static_for<0, nq, 1>{}([&](auto q) {
                                addr[q] = e_addr(number<j0 + decltype(q)::value>{});
                            });
                            __builtin_amdgcn_sched_barrier(0);
                            static_for<0, nq, 1>{}([&](auto q) {
                                issue_e(number<j0 + decltype(q)::value>{}, addr[q]);
                            });
                            // keeps the unguarded stores out of an exec-masked merge block
                            __builtin_amdgcn_sched_barrier(0);
                        }
                        else
                            static_for<0, StoresPerK, 1>{}([&](auto q) {
                                store_e(number<ks * StoresPerK + decltype(q)::value>{}, true);
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
            // Lane holds row 16 * tb + lane % 16, columns 16 * nb + 8 * (lane / 16) + [0, 8). The
            // async stores of the previous tile must have read the stage before it is overwritten.
            s_wait_asynccnt_pinned<0>();
#pragma unroll
            for(int tb = 0; tb < MT; ++tb)
#pragma unroll
                for(int nb = 0; nb < NT; ++nb)
                {
                    union
                    {
                        EDataType e[8];
                        uint4 q;
                    } o;
#pragma unroll
                    for(int e = 0; e < 8; ++e)
                        o.e[e] = type_convert<EDataType>(c[tb][nb][e]);
                    const int r = tb * 16 + lane % 16;
                    stage[r * RowUnits + ((nb * 2 + lane / 16) ^ r % RowUnits)] = o.q;
                }
            e_slab = reinterpret_cast<char*>(Base::ETile(kargs, m0_0 + rw0, m1, n0));
        }
        s_wait_dscnt_pinned<0>();
        static_for<0, Stores, 1>{}([&](auto j) { store_e(j, !full_slab); });
        // The stage must stay allocated until the async stores have read it.
        s_wait_asynccnt_pinned<0>();
#else
        ignore = kargs;
        ignore = m0_0;
        ignore = m1;
        ignore = rows;
        ignore = p_smem;
#endif
    }
};

} // namespace ck_tile
